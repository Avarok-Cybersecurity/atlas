// SPDX-License-Identifier: AGPL-3.0-only
//
// Minimal one-sided RDMA READ shim over libibverbs (RoCEv2), for the expert
// weight tier.
//
// `ibv_post_send` / `ibv_poll_cq` are `static inline` in <infiniband/verbs.h>,
// so they cannot be FFI'd directly from Rust — they must be called from a C
// translation unit. This shim is that unit: it wraps the whole RC-QP lifecycle
// (create -> INIT -> RTR -> RTS), MR registration, one-sided READ posting, and
// completion polling behind a tiny C ABI the Rust side (`verbs.rs`) binds.
//
// Design: one `rs_conn` == one device context + PD + CQ + RC QP, plus a small
// fixed table of registered MRs. Both peer (server, registers its store with
// REMOTE_READ) and client (registers its pinned arena with LOCAL_WRITE) create
// one. QP identity (qpn/psn/gid) is exchanged out-of-band over the existing TCP
// control channel; `rs_connect` drives the state machine with the remote params.
// One-sided: the client is the requester (posts READs); the server QP only has
// to reach RTS to respond, its CPU stays idle (the point of one-sided reads).

#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RS_MAX_MR 512

struct rs_conn {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    union ibv_gid gid;
    struct ibv_mr *mrs[RS_MAX_MR];
    int n_mr;
    uint8_t port;
    int gid_idx;
};

void rs_destroy(struct rs_conn *c);

// Open device `dev_name` (port 1), read the GID at `gid_idx`, allocate PD + CQ,
// create an RC QP and move it to INIT. Returns NULL on any failure.
struct rs_conn *rs_create(const char *dev_name, int gid_idx) {
    struct ibv_device **list = ibv_get_device_list(NULL);
    if (!list) {
        return NULL;
    }
    struct ibv_device *dev = NULL;
    for (int i = 0; list[i]; i++) {
        if (strcmp(ibv_get_device_name(list[i]), dev_name) == 0) {
            dev = list[i];
            break;
        }
    }
    if (!dev) {
        ibv_free_device_list(list);
        return NULL;
    }
    struct rs_conn *c = calloc(1, sizeof(*c));
    if (!c) {
        ibv_free_device_list(list);
        return NULL;
    }
    c->port = 1;
    c->gid_idx = gid_idx;
    c->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (!c->ctx) {
        goto err;
    }
    if (ibv_query_gid(c->ctx, c->port, gid_idx, &c->gid)) {
        goto err;
    }
    c->pd = ibv_alloc_pd(c->ctx);
    if (!c->pd) {
        goto err;
    }
    c->cq = ibv_create_cq(c->ctx, 256, NULL, NULL, 0);
    if (!c->cq) {
        goto err;
    }
    struct ibv_qp_init_attr qa;
    memset(&qa, 0, sizeof(qa));
    qa.send_cq = c->cq;
    qa.recv_cq = c->cq;
    qa.qp_type = IBV_QPT_RC;
    qa.cap.max_send_wr = 256;
    qa.cap.max_recv_wr = 16;
    qa.cap.max_send_sge = 1;
    qa.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qa);
    if (!c->qp) {
        goto err;
    }
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = c->port;
    // Grant REMOTE_READ + REMOTE_WRITE on the QP. The expert tier only issues
    // READs, and its MRs are REMOTE_READ-only, so a stray write is still refused
    // at the MR level — but the KV overflow blade's QP must accept incoming
    // RDMA WRITEs. QP-level flags are a ceiling; per-MR flags do the enforcing.
    attr.qp_access_flags =
        IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(c->qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS)) {
        goto err;
    }
    return c;
err:
    rs_destroy(c);
    return NULL;
}

uint32_t rs_qpn(struct rs_conn *c) { return c->qp->qp_num; }

// Copy this QP's 16-byte GID (the one at gid_idx) out for the wire handshake.
void rs_gid(struct rs_conn *c, uint8_t out[16]) { memcpy(out, c->gid.raw, 16); }

// Register `[addr, addr+len)` as an MR. `flags` is a bitmask giving each side
// exactly the access it needs and no more:
//   bit 0 (1) -> REMOTE_READ   (expert store; also the KV blade's read-back)
//   bit 1 (2) -> REMOTE_WRITE  (KV overflow blade; implies LOCAL_WRITE per the
//                               verbs rule that REMOTE_WRITE requires LOCAL_WRITE)
//   flags == 0 -> LOCAL_WRITE  (the client's landing/bounce buffer)
// REMOTE_READ alone deliberately omits LOCAL_WRITE (requesting it on a PROT_READ
// mmap makes ibv_reg_mr fail — the expert store is registered read-only this way).
// So: expert store = 1, client bounce = 0, KV blade = 3 (RR|RW|LW).
int rs_reg_mr(struct rs_conn *c, void *addr, size_t len, int flags,
              uint32_t *lkey, uint32_t *rkey) {
    if (c->n_mr >= RS_MAX_MR) {
        return -1;
    }
    int access = 0;
    if (flags & 1) {
        access |= IBV_ACCESS_REMOTE_READ;
    }
    if (flags & 2) {
        access |= IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
    }
    if (access == 0) {
        access = IBV_ACCESS_LOCAL_WRITE;
    }
    struct ibv_mr *mr = ibv_reg_mr(c->pd, addr, len, access);
    if (!mr) {
        return -1;
    }
    c->mrs[c->n_mr++] = mr;
    *lkey = mr->lkey;
    *rkey = mr->rkey;
    return 0;
}

// Drive INIT -> RTR -> RTS with the remote QP's params. `remote_psn` is the
// remote's send PSN (our rq_psn); `local_psn` is our send PSN (our sq_psn). The
// path MTU is taken from the port's active MTU. Returns 0, or a negative code
// identifying the failed transition.
int rs_connect(struct rs_conn *c, uint32_t remote_qpn, uint32_t remote_psn,
               uint32_t local_psn, const uint8_t remote_gid[16]) {
    struct ibv_port_attr pa;
    if (ibv_query_port(c->ctx, c->port, &pa)) {
        return -1;
    }
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = pa.active_mtu;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = remote_psn;
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = c->port;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.sgid_index = c->gid_idx;
    attr.ah_attr.grh.traffic_class = 0;
    memcpy(attr.ah_attr.grh.dgid.raw, remote_gid, 16);
    if (ibv_modify_qp(c->qp, &attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        return -2;
    }
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = local_psn;
    attr.max_rd_atomic = 16;
    if (ibv_modify_qp(c->qp, &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC)) {
        return -3;
    }
    return 0;
}

// Post a single one-sided RDMA READ: pull `len` bytes from the remote
// `remote_addr` (protected by `rkey`) into local `local_addr` (registered under
// `lkey`). Signaled, so it generates a completion tagged with `wr_id`.
int rs_post_read(struct rs_conn *c, void *local_addr, uint32_t lkey,
                 uint64_t remote_addr, uint32_t rkey, uint32_t len,
                 uint64_t wr_id) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)local_addr;
    sge.length = len;
    sge.lkey = lkey;
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;
    struct ibv_send_wr *bad = NULL;
    return ibv_post_send(c->qp, &wr, &bad);
}

// Post a single one-sided RDMA WRITE: push `len` bytes from local `local_addr`
// (registered under `lkey`) to the remote `remote_addr` (protected by `rkey`).
// Signaled, so it generates a completion tagged with `wr_id`. Used by the KV
// overflow tier to offload a K/V group into the peer's registered RAM blade.
int rs_post_write(struct rs_conn *c, void *local_addr, uint32_t lkey,
                  uint64_t remote_addr, uint32_t rkey, uint32_t len,
                  uint64_t wr_id) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)local_addr;
    sge.length = len;
    sge.lkey = lkey;
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;
    struct ibv_send_wr *bad = NULL;
    return ibv_post_send(c->qp, &wr, &bad);
}

// Blocking busy-poll for exactly one completion. On success returns 0 and writes
// the completed work-request's id to *out_wr_id; on a completion error returns
// the positive `ibv_wc_status`; on a poll error returns -1.
int rs_poll(struct rs_conn *c, uint64_t *out_wr_id) {
    struct ibv_wc wc;
    for (;;) {
        int n = ibv_poll_cq(c->cq, 1, &wc);
        if (n < 0) {
            return -1;
        }
        if (n == 0) {
            continue;
        }
        *out_wr_id = wc.wr_id;
        if (wc.status != IBV_WC_SUCCESS) {
            return (int)wc.status;
        }
        return 0;
    }
}

void rs_destroy(struct rs_conn *c) {
    if (!c) {
        return;
    }
    if (c->qp) {
        ibv_destroy_qp(c->qp);
    }
    for (int i = 0; i < c->n_mr; i++) {
        if (c->mrs[i]) {
            ibv_dereg_mr(c->mrs[i]);
        }
    }
    if (c->cq) {
        ibv_destroy_cq(c->cq);
    }
    if (c->pd) {
        ibv_dealloc_pd(c->pd);
    }
    if (c->ctx) {
        ibv_close_device(c->ctx);
    }
    free(c);
}
