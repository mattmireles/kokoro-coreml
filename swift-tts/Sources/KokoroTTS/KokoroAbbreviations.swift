import Foundation

/// Tokens ending in a period that bind to the word after them.
///
/// A chunk seam between an abbreviation and the word it modifies ("Dr." /
/// "Hartman", "J. K." / "Rowling") synthesizes as a jarring mid-name pause,
/// because each chunk is voiced with its own sentence-final intonation.
///
/// This mirrors `packages/protocol/shared/kokoro-abbreviations.js` in the Gist
/// repo, which the web and extension chunkers share. The long tail is carried on
/// purpose: an entry costs a few bytes, a false negative costs audibly wrong
/// prosody, and users paste anything from the internet — academic, biblical,
/// military, finance, and science text all show up.
enum KokoroAbbreviations {
    /// Abbreviations that must not end a sentence or a chunk.
    ///
    /// Multi-word source entries such as "Lt. Col." are absent by design: the
    /// boundary check tokenizes on whitespace, so each component is matched
    /// independently through its single-word form.
    static let tokens: Set<String> = [
        "Mr.", "Mrs.", "Ms.", "Mx.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "Ste.", "Hon.",
        "Rev.", "Fr.", "Br.", "Sis.", "Msgr.", "Sgt.", "Lt.", "Capt.", "Col.", "Gen.",
        "Maj.", "Cpl.", "Pvt.", "Cmdr.", "Adm.", "Pres.", "V.P.", "Gov.", "Sen.", "Rep.",
        "Amb.", "Esq.", "Atty.", "Mme.", "Mlle.", "Mons.", "Sra.", "Srta.", "Messrs.",
        "Mmes.", "Ave.", "Blvd.", "Rd.", "Ln.", "Ct.", "Pl.", "Pkwy.", "Hwy.", "Fwy.",
        "Tpke.", "Cir.", "Sq.", "Ter.", "Trl.", "Apt.", "Fl.", "Bldg.", "Rm.", "Mt.",
        "Mtn.", "Ft.", "Pt.", "Hts.", "Vlg.", "P.O.", "No.", "Nos.", "N.", "S.", "E.", "W.",
        "N.E.", "N.W.", "S.E.", "S.W.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.",
        "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "Mon.", "Tue.", "Tues.", "Wed.",
        "Weds.", "Thu.", "Thur.", "Thurs.", "Fri.", "Sat.", "Sun.", "e.g.", "i.e.", "etc.",
        "viz.", "cf.", "ibid.", "id.", "q.v.", "N.B.", "n.b.", "c.", "ca.", "circ.",
        "circa.", "approx.", "vs.", "v.", "al.", "supra.", "infra.", "a.k.a.", "a/k/a.",
        "w/o.", "w/.", "Q.E.D.", "P.S.", "P.P.S.", "vol.", "vols.", "ch.", "chap.",
        "chaps.", "para.", "paras.", "sec.", "secs.", "fig.", "figs.", "p.", "pp.", "pg.",
        "pgs.", "ed.", "eds.", "edn.", "trans.", "transl.", "suppl.", "rev.", "bk.", "bks.",
        "art.", "arts.", "col.", "cols.", "n.", "nn.", "l.", "ll.", "ff.", "f.", "a.m.",
        "p.m.", "A.M.", "P.M.", "A.D.", "B.C.", "B.C.E.", "C.E.", "A.H.", "d.", "b.", "r.",
        "fl.", "Inc.", "Corp.", "Co.", "Ltd.", "L.L.C.", "L.L.P.", "L.P.", "P.C.", "P.A.",
        "P.L.L.C.", "Bros.", "Cos.", "Hldgs.", "Assoc.", "Assn.", "Dept.", "Div.", "Mfg.",
        "Mfr.", "Mgmt.", "Mgr.", "Asst.", "Admin.", "Sec.", "Treas.", "Op.", "Univ.",
        "Inst.", "Comm.", "Soc.", "Fed.", "Co-op.", "Org.", "Intl.", "Natl.", "Nat'l.",
        "U.S.", "U.S.A.", "U.K.", "U.N.", "L.A.", "N.Y.", "D.C.", "N.Y.C.", "U.S.S.R.",
        "E.U.", "U.A.E.", "R.O.K.", "D.P.R.K.", "P.R.C.", "R.O.C.", "G.B.", "Ala.", "Ariz.",
        "Ark.", "Calif.", "Cal.", "Colo.", "Conn.", "Del.", "Fla.", "Ga.", "Ill.", "Ind.",
        "Kan.", "Kans.", "Ky.", "La.", "Mass.", "Md.", "Mich.", "Minn.", "Miss.", "Mo.",
        "Mont.", "Neb.", "Nebr.", "Nev.", "N.H.", "N.J.", "N.M.", "N.Mex.", "N.C.", "N.D.",
        "N.Dak.", "Okla.", "Oreg.", "Or.", "Pa.", "Penn.", "Penna.", "R.I.", "S.C.", "S.D.",
        "S.Dak.", "Tenn.", "Tex.", "Vt.", "Va.", "Wash.", "W.Va.", "Wis.", "Wisc.", "Wyo.",
        "Amer.", "Brit.", "Can.", "Ger.", "Jpn.", "Mex.", "Russ.", "Sp.", "Swed.", "Eng.",
        "Ital.", "Chin.", "Aust.", "Austrl.", "Eur.", "Afr.", "ft.", "in.", "yd.", "mi.",
        "mm.", "cm.", "m.", "km.", "lb.", "lbs.", "oz.", "gr.", "kg.", "qt.", "pt.", "gal.",
        "tsp.", "tbsp.", "Tbsp.", "sq.", "cu.", "hr.", "hrs.", "yr.", "yrs.", "mo.", "mos.",
        "wk.", "wks.", "dia.", "diam.", "max.", "min.", "temp.", "qty.", "wt.", "dim.",
        "est.", "II.", "III.", "IV.", "VI.", "VII.", "VIII.", "IX.", "XI.", "XII.", "XIII.",
        "XIV.", "XV.", "XVI.", "XVII.", "XVIII.", "XIX.", "XX.", "Ph.D.", "M.D.", "D.D.S.",
        "D.V.M.", "D.O.", "J.D.", "M.A.", "M.S.", "M.Sc.", "M.B.A.", "B.A.", "B.S.",
        "B.Sc.", "B.S.N.", "M.S.N.", "R.N.", "L.P.N.", "D.D.", "D.Min.", "Th.D.", "M.Div.",
        "LL.B.", "LL.M.", "LL.D.", "Ed.D.", "D.Phil.", "Litt.D.", "D.M.D.", "Pharm.D.",
        "Psy.D.", "D.Sc.", "D.Eng.", "M.Eng.", "B.Eng.", "M.F.A.", "B.F.A.", "M.P.H.",
        "M.P.A.", "M.S.W.", "A.A.", "A.S.", "G.E.D.", "R.S.V.P.", "R.I.P.", "S.O.S.",
        "A.W.O.L.", "F.Y.I.", "E.T.A.", "E.T.D.", "A.S.A.P.", "C.O.D.", "D.I.Y.",
        "R.O.T.C.", "Exod.", "Lev.", "Num.", "Deut.", "Josh.", "Judg.", "Sam.", "Kgs.",
        "Chr.", "Chron.", "Ezr.", "Neh.", "Est.", "Ps.", "Pss.", "Prov.", "Eccl.", "Isa.",
        "Jer.", "Lam.", "Ezek.", "Dan.", "Hos.", "Obad.", "Jon.", "Mic.", "Nah.", "Hab.",
        "Zeph.", "Hag.", "Zech.", "Mal.", "Matt.", "Mk.", "Lk.", "Jn.", "Rom.", "Cor.",
        "Gal.", "Eph.", "Phil.", "Thess.", "Tim.", "Tit.", "Phlm.", "Heb.", "Jas.", "Pet.",
        "Apoc.", "O.T.", "N.T.",
    ]

    /// Returns whether a whitespace-delimited token must keep the next word attached.
    ///
    /// - Parameter token: Token immediately before a candidate break.
    /// - Returns: True when breaking after this token would split a name or phrase.
    static func isProtectedBoundaryToken(_ token: String) -> Bool {
        tokens.contains(token) || isMultiInitialAcronym(token) || isSingleInitial(token)
    }

    /// Returns whether a token is an enumerated acronym such as `F.B.I.` or `U.S.A.F.`.
    ///
    /// Mirrors the JS `MULTI_INITIAL_ACRONYM` regex `^(?:[A-Z]\.){2,}$`, which is
    /// anchored so ordinary words ending in a period never match.
    ///
    /// - Parameter token: Token immediately before a candidate break.
    /// - Returns: True for two or more capital-plus-period pairs.
    static func isMultiInitialAcronym(_ token: String) -> Bool {
        let characters = Array(token)
        guard characters.count >= 4, characters.count.isMultiple(of: 2) else {
            return false
        }
        var index = 0
        while index < characters.count {
            guard isASCIICapital(characters[index]), characters[index + 1] == "." else {
                return false
            }
            index += 2
        }
        return true
    }

    /// Returns whether a token is a single initial such as `J.` inside a name.
    ///
    /// - Parameter token: Token immediately before a candidate break.
    /// - Returns: True for one capital letter followed by a period.
    static func isSingleInitial(_ token: String) -> Bool {
        let characters = Array(token)
        return characters.count == 2 && isASCIICapital(characters[0]) && characters[1] == "."
    }

    /// Returns whether a character is an ASCII capital letter.
    ///
    /// The JS source matches `[A-Z]`, so accented capitals deliberately do not
    /// count as initials here either.
    ///
    /// - Parameter character: Character to test.
    /// - Returns: True for `A` through `Z`.
    private static func isASCIICapital(_ character: Character) -> Bool {
        character.isASCII && character.isUppercase
    }
}
