#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only liveness of an exact campaign-owned server during its boot gate."""
import atexit
import os
import select

from process_launch_proc import (capture, marker_matches, process_stat, read_owner,
                                 require_linux)


class ProcessFailure(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


class ProcessGuard:
    def __init__(self, record):
        self.descriptor = None
        try:
            require_linux()
            self.owner = read_owner(record)
            try:
                self.descriptor = os.pidfd_open(self.owner['pid'])
            except ProcessLookupError:
                raise ProcessFailure('process-exited', 'owned server is already gone') from None
            self.check()
            capture(self.owner)  # Prove the original launch, not merely a live PID.
            self.check()
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            self.close()
            raise ProcessFailure('process-ownership-unproven', str(error)) from error
        except ProcessFailure:
            self.close()
            raise
        atexit.register(self.close)

    def close(self):
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None

    def check(self):
        try:
            if select.select([self.descriptor], [], [], 0)[0]:
                raise ProcessFailure('process-exited', 'owned server exited before boot completed')
            current = process_stat(self.owner['pid'])
            if current['state'] in ('Z', 'X'):
                raise ProcessFailure('process-exited', 'owned server is no longer running')
            if (any(current[key] != self.owner[key] for key in
                    ('pid', 'start_ticks', 'pgid', 'sid')) or
                    not marker_matches(self.owner['pid'], self.owner['run_marker'])):
                raise ValueError('process identity or ownership marker changed during boot')
        except (FileNotFoundError, ProcessLookupError):
            raise ProcessFailure('process-exited', 'owned server disappeared during boot') from None
        except (OSError, ValueError) as error:
            raise ProcessFailure('process-ownership-unproven', str(error)) from error
