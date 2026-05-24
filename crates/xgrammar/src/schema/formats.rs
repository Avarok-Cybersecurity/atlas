// SPDX-License-Identifier: AGPL-3.0-only
//
// JSON Schema `format` -> regex mapping — port of
// `JSONSchemaConverter::JSONFormatToRegexPattern` from
// `cpp/json_schema_converter.cc`.
//
// Returns the regex (as used by the regex->EBNF converter) for the
// supported string `format` keywords, or `None` for unknown formats.

/// Resolve a JSON Schema `format` keyword to its regex pattern.
/// Unknown formats return `None` (the converter then falls back to a
/// plain string). Patterns are byte-for-byte the C++ `regex_map`.
pub fn format_to_regex(format: &str) -> Option<String> {
    let atext = r"[\w!#$%&'*+/=?^`{|}~-]";
    let dot_string = format!(r"({atext}+(\.{atext}+)*)");
    let quoted_string = r#"\\"(\\[\x20-\x7E]|[\x20\x21\x23-\x5B\x5D-\x7E])*\\""#;
    let domain =
        r"([A-Za-z0-9]([\-A-Za-z0-9]*[A-Za-z0-9])?)((\.[A-Za-z0-9][\-A-Za-z0-9]*[A-Za-z0-9])*)";

    let pat: String = match format {
        "email" => format!("^({dot_string}|{quoted_string})@{domain}$"),
        "date" => r"^(\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\d|3[01]))$".to_string(),
        "time" => {
            r"^([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)(\.\d+)?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)$"
                .to_string()
        }
        "date-time" => {
            r"^(\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\d|3[01]))T([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)(\.\d+)?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)$"
                .to_string()
        }
        "duration" => {
            r"^P((\d+D|\d+M(\d+D)?|\d+Y(\d+M(\d+D)?)?)(T(\d+S|\d+M(\d+S)?|\d+H(\d+M(\d+S)?)?))?|T(\d+S|\d+M(\d+S)?|\d+H(\d+M(\d+S)?)?)|\d+W)$"
                .to_string()
        }
        "ipv4" => {
            let decbyte = r"(25[0-5]|2[0-4]\d|[0-1]?\d?\d)";
            format!(r"^({decbyte}\.){{3}}{decbyte}$")
        }
        "ipv6" => ipv6_pattern(),
        "hostname" => {
            r"^([a-z0-9]([a-z0-9-]*[a-z0-9])?)(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$".to_string()
        }
        "uuid" => {
            r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
                .to_string()
        }
        "uri" => uri_pattern(),
        "uri-reference" => uri_reference_pattern(),
        "uri-template" => uri_template_pattern(),
        "json-pointer" => {
            r"^(/([\x00-\x2E]|[\x30-\x7D]|[\x7F-\u{10FFFF}]|~[01])*)*$".to_string()
        }
        "relative-json-pointer" => {
            r"^(0|[1-9][0-9]*)(#|(/([\x00-\x2E]|[\x30-\x7D]|[\x7F-\u{10FFFF}]|~[01])*)*)$"
                .to_string()
        }
        _ => return None,
    };
    Some(pat)
}

fn ipv6_pattern() -> String {
    [
        "(",
        r"([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|",
        r"([0-9a-fA-F]{1,4}:){1,7}:|",
        r"([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|",
        r"([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|",
        r"([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|",
        r"([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|",
        r"([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|",
        r"[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|",
        r":((:[0-9a-fA-F]{1,4}){1,7}|:)|",
        r"::(ffff(:0{1,4}){0,1}:){0,1}",
        r"((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}",
        r"(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|",
        r"([0-9a-fA-F]{1,4}:){1,4}:",
        r"((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}",
        r"(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])",
        ")",
    ]
    .concat()
}

fn uri_pattern() -> String {
    let schema_pat = r"[a-zA-Z][a-zA-Z+\.-]*";
    let pchar = r"([\w\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])";
    let qfc = r"([\w\.~!$&'()*+,;=:@/\?-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    let query = format!(r"(\?{qfc})?");
    let fragment = format!("(#{qfc})?");
    let path_abempty = format!("(/{pchar}*)*");
    let path_absolute_rootless_empty = format!("/?({pchar}+(/{pchar}*)*)?");
    let userinfo = r"([\w\.~!$&'()*+,;=:-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    let host = r"([\w\.~!$&'()*+,;=-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    let authority = format!(r"({userinfo}@)?{host}(:\d*)?");
    let hier_part = format!("(//{authority}{path_abempty}|{path_absolute_rootless_empty})");
    format!("^{schema_pat}:{hier_part}{query}{fragment}$")
}

fn uri_reference_pattern() -> String {
    let pchar = r"([\w\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])";
    let qfc = r"([\w\.~!$&'()*+,;=:@/\?-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    let query = format!(r"(\?{qfc})?");
    let fragment = format!("(#{qfc})?");
    let path_abempty = format!("(/{pchar}*)*");
    let path_absolute = format!("/({pchar}+(/{pchar}*)*)?");
    let segment_nz_nc = r"([\w\.~!$&'()*+,;=@-]|%[0-9A-Fa-f][0-9A-Fa-f])+";
    let path_noscheme = format!("{segment_nz_nc}(/{pchar}*)*");
    let userinfo = r"([\w\.~!$&'()*+,;=:-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    let host = r"([\w\.~!$&'()*+,;=-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    let authority = format!(r"({userinfo}@)?{host}(:\d*)?");
    let relative_part = format!("(//{authority}{path_abempty}|{path_absolute}|{path_noscheme})?");
    format!("^{relative_part}{query}{fragment}$")
}

fn uri_template_pattern() -> String {
    let literals =
        r"([\x21\x23-\x24\x26\x28-\x3B\x3D\x3F-\x5B\x5D\x5F\x61-\x7A\x7E]|%[0-9A-Fa-f][0-9A-Fa-f])";
    let op = r"[+#\./;\?&=,!@|]";
    let varchar = r"(\w|%[0-9A-Fa-f][0-9A-Fa-f])";
    let varname = format!(r"{varchar}(\.?{varchar})*");
    let varspec = format!(r"{varname}(:[1-9]\d?\d?\d?|\*)?");
    let variable_list = format!("{varspec}(,{varspec})*");
    let expression = format!(r"\{{({op})?{variable_list}\}}");
    format!("^({literals}|{expression})*$")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_formats_resolve() {
        assert!(format_to_regex("email").is_some());
        assert!(format_to_regex("date").is_some());
        assert!(format_to_regex("uuid").is_some());
        assert!(format_to_regex("ipv4").is_some());
        assert!(format_to_regex("uri").is_some());
        assert!(format_to_regex("date-time").is_some());
    }

    #[test]
    fn unknown_format_is_none() {
        assert!(format_to_regex("not-a-format").is_none());
    }

    #[test]
    fn date_pattern_exact() {
        assert_eq!(
            format_to_regex("date").unwrap(),
            r"^(\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\d|3[01]))$"
        );
    }
}
