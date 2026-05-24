// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNF token type definitions. Split out of `lexer.rs` to keep each
// file under the 250-line cap. Port of `EBNFLexer::TokenType` and
// `EBNFLexer::Token` from xgrammar `cpp/grammar_parser.h`.

/// EBNF token kinds. Port of `EBNFLexer::TokenType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    /// Name on the left of `::=`, e.g. `root`.
    RuleName,
    /// A reference to a rule, or a macro name (e.g. `TagDispatch`).
    Identifier,
    /// A `"..."` string literal.
    StringLiteral,
    /// `true` / `false`.
    BooleanLiteral,
    /// An integer literal.
    IntegerLiteral,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `|`
    Pipe,
    /// `,`
    Comma,
    /// End of input.
    EndOfFile,
    /// `::=`
    Assign,
    /// `=`
    Equal,
    /// `*`
    Star,
    /// `+`
    Plus,
    /// `?`
    Question,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `-` inside a character class.
    Dash,
    /// `^` inside a character class.
    Caret,
    /// A single (possibly escaped) character inside a character class.
    CharInCharClass,
    /// A regex-style escape with special meaning inside a class (`\d`…).
    EscapeInCharClass,
    /// `(=` — opens a lookahead assertion.
    LookaheadLParen,
}

/// The processed value carried by a [`Token`].
#[derive(Debug, Clone, PartialEq)]
pub enum TokenValue {
    /// No processed value.
    None,
    /// String value (string literals; identifier names; class escapes).
    Str(String),
    /// Integer value (integer literals).
    Int(i64),
    /// Boolean value (`true`/`false`).
    Bool(bool),
    /// Codepoint value (a character inside a character class).
    Codepoint(i32),
}

/// A single lexer token. Port of `EBNFLexer::Token`.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The token kind.
    pub ty: TokenType,
    /// Original source text of the token.
    pub lexeme: String,
    /// Processed value.
    pub value: TokenValue,
    /// 1-based source line.
    pub line: i32,
    /// 1-based source column.
    pub column: i32,
}

/// A lexer error with source position.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("EBNF lexer error at line {line}, column {column}: {msg}")]
pub struct LexError {
    /// 1-based source line.
    pub line: i32,
    /// 1-based source column.
    pub column: i32,
    /// Human-readable error message.
    pub msg: String,
}
