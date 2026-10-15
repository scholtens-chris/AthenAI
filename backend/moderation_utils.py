import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

# Screens user prompts before anything reaches retrieval or the model and decides how to handle them:
#   allow   - send to the model, optionally with trusted guidance notes (e.g. "quoting a slur for history").
#   respond - reply with a fixed, supportive message instead of a model answer (e.g. self-harm, other users' data).
#   decline - briefly refuse a request that would facilitate harm, and offer relevant help.
#
# The screen is deliberately narrow. Study material covers discrimination, sexuality, health, violence, and
# history, so clinical, academic, quoted, and help-seeking wording is allowed; rough language and missing
# manners are never a reason to refuse; and polite wording never makes a harmful request acceptable.
# Regexes cannot read intent, so anything ambiguous goes to the model with safety guidance (behavior_policy.py).

VOWELS = "aeiou"
# Leetspeak substitutions, applied only when the character touches a letter ("sh1t", "@ss").
LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
LEET_CHARS = re.compile(r"(?<=[a-z])[013457@$]|[013457@$](?=[a-z])")
LEET_BANG = re.compile(r"(?<=[a-z])!(?=[a-z])")
# Phrases that contain a flagged word but are harmless.
ALLOWED_PHRASES = re.compile(r"\bchinks? in (?:the|its|his|her|their|our|my|your) armou?r\b")


def term_pattern(term: str) -> str:
    """Regex for a word that tolerates repeated letters, masked vowels, and dotted spelling ("f*ck", "s.h.i.t")."""
    parts = []
    for char in term:
        if char == " ":
            parts.append(r"[\W_]+")
        elif char in VOWELS:
            parts.append(f"(?:{char}|\\*)+")
        else:
            parts.append(f"{re.escape(char)}+")
    return r"[._*-]?".join(parts)


def word_list_regex(terms: tuple[str, ...], prefixes: str = "", suffixes: str = "") -> re.Pattern:
    body = "|".join(term_pattern(term) for term in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<![a-z])(?:{prefixes})?(?:{body})(?:{suffixes})?(?![a-z])")


SLUR_REGEX = word_list_regex(
    (
        "nigger", "nigga", "niggah", "sand nigger", "chink", "gook", "spic", "kike", "wetback",
        "beaner", "raghead", "towelhead", "jigaboo", "porch monkey", "paki", "faggot", "fag",
        "tranny", "shemale", "retard",
    ),
    suffixes=r"s|z",
)
# A slur being discussed rather than aimed at someone: quoted, named as a word, taken from a text, or reported.
SLUR_DISCUSSION = re.compile(
    r"\b(?:the (?:word|term|slur|phrase|label|epithet|insult)s?|n-word|f-word|slurs?|quot(?:e|es|ed|ing|ation)|"
    r"said|says|saying|wrote|writes|written|used|uses|using|use of|usage|mean|means|meaning|definition|define|"
    r"etymology|origins?|history|historical|historically|offensive|derogatory|reclaim(?:ed|ing)?|"
    r"censor(?:ed|ing|ship)?|lyrics|novel|book|poem|speech|lecture|transcript|film|movie|song|author|"
    r"character|called me|calls me|calling me|call me|yelled|shouted|texted|posted|report(?:ed|ing)?)\b"
)
QUOTE_CHARS = "\"“”'`‘’«»"

PROFANITY_REGEX = word_list_regex(
    ("fuck", "fuk", "fck", "phuck", "fvck", "shit", "cunt", "bitch", "asshole", "dickhead", "twat", "wanker"),
    prefixes=r"mother|mutha|bull|dumb|horse|dip",
    suffixes=r"s|es|ed|er|ers|ing|in|face|head|heads|wit|tard|y",
)
PROFANE_ABBREVIATIONS = re.compile(r"(?<![a-z])(?:wtf|stfu|gtfo|fml)(?![a-z])")

YOU = r"(?:you|u|ya|athen ?ai|athena)"
INSULTS = (
    r"(?:stupid|dumb|idiot(?:ic)?|moron(?:ic)?|useless|worthless|pathetic|trash|garbage|loser|retarded|"
    r"brainless|imbecile|dumbass|clown|a joke|incompetent|annoying|terrible|the worst)"
)
ABUSE_REGEX = re.compile(
    r"|".join(
        (
            rf"\b{YOU}(?:\s+(?:are|r|is)|'re|re)\s+(?:so |such |an? |really |very |totally |the |fucking )*{INSULTS}\b",
            r"\b(?:you|u) (?:(?:stupid|dumb|useless|worthless|pathetic|fucking) )?(?:idiot|moron|imbecile|loser|clown|dumbass)\b",
            r"\b(?:stupid|dumb|useless|worthless|idiot|moron|pathetic) (?:bot|ai|owl|machine|app|assistant|computer)\b",
            rf"\b{YOU} suck\b",
            rf"\bi hate {YOU}\b",
            r"\b(?:shut up|screw (?:you|u|off)|piss off|go to hell|drop dead|go die|kill yourself|kys|eat shit)\b",
            rf"\b(?:fuck|f\*+ck|f u) (?:{YOU}|off)\b",
        )
    )
)

GROUP_ADJECTIVES = (
    r"(?:black|white|brown|asian|jewish|muslim|arab|mexican|latino|latina|hispanic|chinese|indian|african|"
    r"gay|lesbian|bisexual|trans|transgender|queer|disabled|immigrant|christian|hindu|sikh|catholic|"
    r"mormon|native|indigenous)"
)
GROUPS = (
    rf"(?:{GROUP_ADJECTIVES} (?:people|persons|folks|men|women|guys|kids|ones|community)|"
    r"blacks|whites|jews|muslims|arabs|mexicans|latinos|latinas|hispanics|asians|indians|africans|gays|"
    r"lesbians|queers|transgenders?|transsexuals|homosexuals|immigrants|migrants|refugees|foreigners|"
    r"women|girls|females|christians|hindus|sikhs|catholics|mormons|atheists|lgbtq?\+?|the lgbtq?\+? community)"
)
HOSTILE = (
    r"(?:inferior|subhuman|sub human|animals|vermin|savages|disgusting|evil|stupid|dumb|criminals|"
    r"terrorists|parasites|a disease|diseased|trash|garbage|filth|filthy|scum|cockroaches|rats|monkeys|"
    r"apes|less than human|not human|not real people|abominations?|freaks|degenerates?|a mental illness|"
    r"perverts|pedophiles|groomers|lazy|thieves|rapists|worthless|a plague|a cancer|unnatural|"
    r"ruining (?:this|our|the) country)"
)
LGBTQ_IDENTITY = r"(?:gay|lesbian|bisexual|trans|transgender|queer|homosexual)"
# A call to violence needs a speaker voicing it, so "why did the Nazis kill Jews" stays allowed.
VIOLENCE_SPEAKER = (
    r"(?:^|[.!?] |\bwe (?:should|must|need to|have to) |\blet'?s |\bgonna |"
    r"\bi (?:want|wanna|would like|am going|'m going|will|'ll|would) (?:to )?)"
)
BIGOTRY_REGEX = re.compile(
    r"|".join(
        (
            rf"\b(?:i )?(?:hate|despise|can't stand|cant stand) (?:all |the |those |these )?{GROUPS}\b",
            rf"\b{GROUPS} (?:are|r|is) (?:all |just |so |such |nothing but |a bunch of |basically |literally )*(?:an? )?{HOSTILE}\b",
            rf"{VIOLENCE_SPEAKER}(?:kill|exterminate|gas|lynch|hang|shoot|get rid of|wipe out) (?:all )?(?:the |those |these )?{GROUPS}\b",
            rf"\b{GROUPS} (?:should|must|need to|deserve to|ought to) (?:all )?(?:die|be killed|be shot|be gassed|"
            r"be hanged|be lynched|be deported|be exterminated|be wiped out|burn|not exist|be banned|be locked up|rot)\b",
            rf"\b(?:being )?{LGBTQ_IDENTITY} (?:people )?is (?:a |an )?(?:sin|disease|mental illness|disgusting|wrong|"
            r"unnatural|abomination|sick|perversion)\b",
            r"\bgo back to (?:your|their) (?:own )?(?:country|countries)\b",
        )
    )
)
DENIAL_REGEX = re.compile(
    r"\b(?:the )?holocaust (?:never happened|was a hoax|is a hoax|was fake|is a lie|was a lie|didn't happen|didnt happen)\b"
)
# Reporting, quoting, or questioning a claim is study or help-seeking, not bigotry.
REPORTED_CLAIM = re.compile(
    r"\b(?:says?|said|saying|claim(?:s|ed|ing)?|believed?|argue[sd]?|stereotypes?|myths?|propaganda|portray(?:s|ed)?|"
    r"depict(?:s|ed)?|is it true|is it a myth|why do (?:some )?people|why did|how did|did they|according to|"
    r"the idea that|the belief that|the notion that|told me|tells me|telling me|called me|calls me|yelled|shouted|"
    r"wrote|posted|texted|keeps saying|how (?:do|should|can) i (?:respond|reply|report|answer)|report(?:ed|ing)?|"
    r"debunk|refute|rebut|counter|respond to|is it (?:ok|okay|wrong|racist|sexist|homophobic|transphobic|offensive)|"
    r"quote|quoted)\b|[\"“”«»]"
)

PERSON = (
    r"(?:someone|somebody|anyone|a person|people|him|her|them|my (?:wife|husband|girlfriend|boyfriend|partner|ex|"
    r"mom|mother|dad|father|brother|sister|roommate|classmate|teacher|professor|boss|coworker|neighbou?r|friend|"
    r"baby|child|kid|son|daughter|family|students?|stepdad|stepmom)|(?:a|the|this|that) (?:teacher|professor|cop|"
    r"police officer|student|kid|child|baby|classmate|neighbou?r|president))"
)
HARM_VERBS = r"(?:kill|murder|poison|stab|shoot|strangle|suffocate|drown|hurt|beat up|torture|kidnap|maim|run over)"
VIOLENCE_PLAN_REGEX = re.compile(
    rf"\b(?:how (?:do|can|could|should|would) i|how to|help me|best way to|easiest way to|ways? to|"
    rf"i (?:want|need|plan|am going|'m going) to|i'?m gonna|i will|i'll) (?:(?!not\b|never\b)[a-z']+ ){{0,2}}?"
    rf"{HARM_VERBS} {PERSON}(?!'s)(?![a-z])"
)

WEAPON_TARGETS = (
    r"(?:pipe ?bombs?|(?<!bath )(?<!seed )(?<!photo )bombs?|explosives?|explosive devices?|ieds?|molotov(?: cocktails?)?|"
    r"napalm|grenades?|detonators?|nerve (?:agents?|gas)|sarin|tabun|novichok|vx(?: gas)?|ricin|anthrax|"
    r"botulinum toxin|mustard gas|chlorine gas|chemical weapons?|bio ?weapons?|biological weapons?|dirty bombs?|"
    r"nuclear (?:weapons?|bombs?)|meth|methamphetamine|crystal meth|fentanyl|crack cocaine|ghost guns?|"
    r"untraceable (?:guns?|firearms?)|(?:gun |firearm )?silencers?|auto sears?|full[- ]auto conversions?)"
)
MAKE_VERBS = (
    r"(?:make|making|build|building|create|creating|assemble|assembling|construct|constructing|synthesi[sz]e|"
    r"synthesi[sz]ing|produce|producing|cook|cooking|manufacture|manufacturing|brew|brewing|3d ?print(?:ing)?|"
    r"weaponi[sz]e|weaponi[sz]ing|extract|extracting|culture|culturing|grow|growing|mix|mixing|rig|rigging)"
)
INSTRUCTION_CUE = (
    r"(?:how (?:do|can|could|would|should|might) (?:i|we|you|someone|somebody|one|a person|people)|how to|how is|"
    r"how are|steps?|step by step|instructions?|recipe|guide|tutorial|walk me through|teach me|show me|tell me|"
    r"explain how|help me|i want to|i need to|i'?m going to|i am going to|i plan to|ways? to|best way|"
    r"easiest way|cheapest way|at home|homemade|diy|materials|ingredients|give me|write)"
)
HISTORICAL_CUE = re.compile(r"\bhow (?:did|were|was)\b|\bwhy did\b|\bhistory of\b|\bwhat happened\b|\bwho (?:made|built|invented)\b")
WEAPON_REQUEST_REGEX = re.compile(
    rf"\b{MAKE_VERBS} (?:(?:me |us )?(?:an? |some |the |my own |your own |our own |homemade |a homemade |more )?)"
    rf"(?:[a-z]+ ){{0,2}}?{WEAPON_TARGETS}(?![a-z])"
    rf"|\b(?:recipes?|instructions?|guides?|tutorials?|blueprints?|schematics?|formulas?) (?:for|to|on) "
    rf"(?:making |building |how to make )?(?:an? |the |some )?(?:[a-z]+ ){{0,2}}?{WEAPON_TARGETS}(?![a-z])"
)
INSTRUCTION_CUE_REGEX = re.compile(rf"\b{INSTRUCTION_CUE}\b")

SELF_HARM_INTENT_REGEX = re.compile(
    r"\b(?:i (?:want|wanna|am going|'m going|plan|intend|am planning|'m planning|have decided|decided) to|"
    r"i'?m gonna|i will|i'll|i should(?:n't)? just|i keep thinking about|thinking about|thinking of) "
    r"(?:kill(?:ing)? myself|end(?:ing)? (?:my life|it all)|commit(?:ting)? suicide|hurt(?:ing)? myself|cut(?:ting)? myself|"
    r"tak(?:e|ing) my (?:own )?life)\b"
    r"|\bi(?:'m| am) (?:feeling )?suicidal\b"
    r"|\bi (?:don'?t|do not) want to (?:live|be alive|exist) anymore\b"
    r"|\b(?:no reason to live|better off dead|better off without me)\b"
)
SELF_HARM_METHOD_REGEX = re.compile(
    r"\bhow (?:to|do i|can i|should i|could i) (?:kill myself|commit suicide|end my life|hang myself|cut myself|"
    r"overdose|od)\b"
    r"|\b(?:painless|quickest|easiest|fastest|surest) way to (?:die|kill myself|end my life|commit suicide)\b"
    r"|\bhow (?:many|much) (?:[a-z0-9]+ ){1,4}(?:to|would|will) (?:kill me|end my life|die|overdose|od)\b"
)

ACCOUNT_OWNER = (
    r"(?:his|her|their|someone'?s|somebody'?s|another (?:person|user|student)'?s|(?:my )?[a-z]+'s|"
    r"my (?:ex|girlfriend|boyfriend|wife|husband|partner|teacher|professor|roommate|friend|boss|classmate|sister|"
    r"brother|mom|dad|mother|father)'?s?)"
)
ACCOUNT_TARGETS = (
    r"(?:accounts?|email|e-mail|inbox|phone|instagram|insta|snapchat|facebook|tiktok|twitter|discord|computer|"
    r"laptop|icloud|gmail|wifi|wi-fi|webcam|messages|dms|texts|grades? portal|network)"
)
ACCOUNT_INTRUSION_REGEX = re.compile(
    rf"\b(?:hack|hacking|break|breaking|get|getting|log|logging|sneak|sneaking) (?:in ?to|into) {ACCOUNT_OWNER} "
    rf"(?:[a-z']+ )?{ACCOUNT_TARGETS}\b"
    rf"|\b(?:hack|hacking) {ACCOUNT_OWNER} (?:[a-z']+ )?{ACCOUNT_TARGETS}\b"
    rf"|\b(?:find out|figure out|get|crack|guess|steal|bypass|reset) {ACCOUNT_OWNER} (?:[a-z']+ )?"
    r"(?:password|passcode|pin|login|credentials|2fa)\b"
    r"|\b(?:track|trace) (?:someone|somebody|my (?:ex|girlfriend|boyfriend|wife|husband|partner)|him|her)(?:'s)? "
    r"(?:location|phone|whereabouts)\b"
    r"|\b(?:spy on|stalk|dox|doxx) (?:someone|somebody|my|him|her|them|a|this|that)\b"
)

OTHER_USERS_DATA_REGEX = re.compile(
    r"\b(?:other|another|different|someone else'?s|everyone'?s|all) (?:users?|students?|people|persons?|person|"
    r"classmates?|accounts?|sessions?)?(?:'s|s')? ?(?:chats?|conversations?|messages?|questions?|prompts?|sessions?|"
    r"uploads?|uploaded files|files|notes|data|chat history|history|information|info|grades?|documents?|activity)\b"
    r"|\bwhat (?:(?:did|have|are|do) )?(?:the )?other (?:users?|students?|people|classmates) (?:have )?(?:ask|asked|"
    r"asking|upload|uploaded|uploading|say|said|search|searched|type|typed)\b"
    r"|\b(?:session|user) (?:id )?[0-9a-f]{8}[0-9a-f-]*'?s? (?:chats?|history|files|data|messages|uploads)\b"
)
OTHER_USERS_ACCESS = re.compile(
    r"\b(?:show|give|tell|see|read|access|view|list|share|send|open|look at|pull up|what (?:did|have|are|do)|"
    r"can you|could you|let me)\b"
)

HARASSMENT_REQUEST_REGEX = re.compile(
    r"\b(?:write|draft|make|create|compose|come up with|help me (?:write|make|draft)|give me) "
    r"(?:me )?(?:an? |some |the )?(?:really |very )?(?:mean|threatening|harass(?:ing)?|insulting|nasty|cruel|"
    r"humiliating|degrading|bullying|hateful|intimidating|racist|sexist|homophobic|transphobic) "
    r"(?:message|text|email|post|comment|note|letter|tweet|dm|reply|roast|joke|rumou?r)s?\b"
    r"|\bhelp me (?:bully|harass|threaten|intimidate|humiliate|blackmail|cyberbully)\b"
)

EXPLICIT_TERMS = (
    r"(?:porn|porno|pornography|hentai|nudes|nude (?:pics?|photos?)|naked (?:pics?|photos?|pictures?|selfies?)|"
    r"nsfw (?:content|pics?|images?|stories?|videos?)|erotica|smut|sex tapes?)"
)
EXPLICIT_REQUEST_REGEX = re.compile(
    r"|".join(
        (
            rf"\b(?:send|show|find|give|get|link|share|write|generate|create|make)(?: me)?"
            rf"(?: (?:some|a|an|the|more|your|good|free|hot))* {EXPLICIT_TERMS}(?![a-z])"
            r"(?! (?:addiction|use|research|stud|effects?|industry|laws?|literacy|consumption|regulation))",
            r"\b(?:talk dirty|dirty talk|sex chat|cyber ?sex|phone sex|strip for me|sext(?:ing)? (?:me|with me))\b",
            rf"\b(?:have sex|sleep|hook up|make out) with (?:me|{YOU})\b",
            r"\b(?:lick|suck) (?:me|my)\b",
            r"\b(?:write|tell|describe|share|give|send)(?: me)? (?:an? |your |some )?(?:sex(?:ual)?|erotic|dirty|smutty|"
            r"steamy|nsfw|explicit|kinky) (?:story|stories|scene|scenes|fanfic|fan fiction|roleplay|role play|"
            r"fantas(?:y|ies)|poem|message|text|chat)\b",
            r"\b(?:let'?s|wanna|want to|can we|could we) (?:do |have )?(?:a |some )?(?:sex(?:ual)?|erotic|dirty|nsfw|kinky) ?"
            r"(?:roleplay|role play|chat|talk)\b",
            r"\b(?:sexy|nude|naked) (?:pics?|photos?|selfies?) of (?:you|yourself)\b",
            r"\b(?:send|show) (?:me )?(?:your )?(?:nudes|boobs|pics of your body)\b",
        )
    )
)
MINOR_TERMS = (
    r"(?:child|children|kids?|minors?|underage|under ?age|preteens?|pre-teens?|toddlers?|little (?:girls?|boys?)|"
    r"(?:1[0-7]|[1-9])[- ](?:year|yr)[- ]olds?|loli|shota|schoolgirls?|teens?|teenagers?)"
)
SEXUAL_TERMS = r"(?:sexy|nude|naked|erotic|porn\w*|sexual(?:ly)?|sex|nsfw|explicit|lewd)"
MINOR_SEXUAL_REGEX = re.compile(
    rf"\b{SEXUAL_TERMS}\b(?:\W+\w+){{0,5}}?\W+{MINOR_TERMS}\b|\b{MINOR_TERMS}\b(?:\W+\w+){{0,5}}?\W+{SEXUAL_TERMS}\b"
    r"|\b(?:child porn\w*|csam|loli|shota)\b"
)
# Protecting children, reporting, education, and disclosure are allowed and go to the model with care notes.
MINOR_PROTECTIVE = re.compile(
    r"\b(?:report\w*|recogni[sz]\w*|signs?|prevent\w*|protect\w*|warning|safety|safe|talk to my|talk to (?:a|my|our) "
    r"(?:kid|child|son|daughter)|educat\w*|statistics?|laws?|legal|research|survivors?|abuse[ds]?|abusive|"
    r"happened to me|groom\w*|consent|age of consent|puberty|development|health|curriculum|detect\w*|moderat\w*)\b"
)

SEXUAL_HEALTH_REGEX = re.compile(
    r"\b(?:sex|sexual(?:ly)?|std|stds|sti|stis|hiv|hpv|herpes|chlamydia|gonorrh?ea|syphilis|condoms?|contracepti\w*|"
    r"birth control|pregnan\w*|menstrua\w*|periods?|erections?|libido|masturbat\w*|jerk(?:ing)? off|consent|orgasms?|"
    r"vagina\w*|penis|plan b|morning after pill|ovulat\w*|porn\w*)\b"
)
MEDICAL_REGEX = re.compile(
    r"\b(?:doses?|dosage|mg|overdos\w*|symptoms?|diagnos\w*|medications?|medicines?|prescri\w*|side effects?|"
    r"chest pain|bleeding|fever|vaccines?|treatments?|ibuprofen|tylenol|acetaminophen|paracetamol|antibiotics?|"
    r"insulin|antidepressants?|lethal dose|ld50|infection|rash|lump)\b"
)
LEGAL_REGEX = re.compile(
    r"\b(?:legal(?:ly)?|illegal|lawyer|attorney|sue|suing|lawsuit|court|contract|lease|landlord|evict\w*|visa|"
    r"immigration status|arrested|copyright|plagiari\w*|academic misconduct|title ix)\b"
)
FINANCIAL_REGEX = re.compile(
    r"\b(?:invest\w*|stocks?|crypto\w*|bitcoin|loans?|debt|credit cards?|credit score|tax(?:es)?|mortgage|"
    r"retirement|401k|savings|financial aid|fafsa)\b"
)
DISTRESS_REGEX = re.compile(
    r"\b(?:want to|wanna|gonna|going to|could just|might as well) die\b|\bkill me now\b|\bi can'?t (?:do this|take it|cope) anymore\b"
    r"|\bhopeless\b|\bworthless\b"
)
SELF_HARM_TOPIC_REGEX = re.compile(
    r"\b(?:suicid\w*|self[- ]harm\w*|self[- ]injur\w*|cutting|overdos\w*|eating disorders?|anorexi\w*|bulimi\w*)\b"
)
CHALLENGE_REGEX = re.compile(
    r"\b(?:you(?:'re| are) (?:wrong|mistaken|incorrect)|that(?:'s| is) (?:wrong|not right|incorrect|not true)|"
    r"(?:not|isn't) (?:right|correct)|fix (?:your|the|that) answer|you got (?:it|that|this) wrong|are you sure|"
    r"i think you(?:'re| are| made)|that's not what)\b"
)
TRAUMA_REGEX = re.compile(
    r"\b(?:abused|abusing me|assaulted|raped|molested|harass(?:ed|ing|ment)|bull(?:ied|ying)|stalk(?:ed|ing)|"
    r"threatened|groomed|trauma\w*|ptsd|domestic violence)\b"
)

# Words that carry no request on their own; used to tell "you're useless" from "you're useless, fix question 2".
FILLER_WORDS = {
    "i", "im", "i'm", "you", "u", "your", "you're", "youre", "are", "r", "is", "so", "such", "the", "a", "an", "this",
    "that", "it", "its", "and", "just", "really", "very", "totally", "honestly", "literally", "omg", "ugh", "lol",
    "bot", "ai", "athenai", "athena", "app", "owl", "seriously", "man", "dude", "bro", "god", "oh", "my", "me", "of",
}

RESPONSES = {
    "harassment": (
        "I won't direct slurs at anyone or help use them as insults. If you're studying how a word like that has "
        "been used, in history, literature, or a lecture, ask about that and I'll help. I'm also glad to help with "
        "anything in your material."
    ),
    "harassment_request": (
        "I won't help write something meant to hurt, threaten, or harass someone. If you're dealing with a "
        "conflict, I can help you say what's bothering you in a way that's direct without being cruel."
    ),
    "bigotry": (
        "I won't help argue that a group of people is inferior or deserves harm. If you're studying where claims "
        "like this come from, how they've been used, or what the evidence actually shows, I'm glad to dig into that."
    ),
    "denial": (
        "The Holocaust is one of the most thoroughly documented events in history. Nazi Germany and its "
        "collaborators murdered about six million Jews, along with millions of others, and the evidence includes "
        "German government records, the camps themselves, survivor and perpetrator testimony, and the Nuremberg "
        "trials. If you're studying denial itself, such as how it spreads and how historians respond to it, I'm "
        "glad to help with that."
    ),
    "explicit": (
        "I don't write sexual content or take part in sexual roleplay. I'm glad to answer sexual-health or "
        "relationship questions factually, or to help with anything in your study material."
    ),
    "minors": (
        "I can't help with anything that sexualizes minors. If you're worried a child is being harmed, contact "
        "local emergency services or a child-protection hotline (in the US, Childhelp at 1-800-422-4453). I can "
        "also explain how to recognize and report abuse."
    ),
    "dangerous": (
        "I can't give instructions for making weapons, explosives, or dangerous drugs. I can explain the "
        "underlying science at a general level, the history, or safety and legal information, if that would help."
    ),
    "violence": (
        "I can't help plan harm to someone. If you're in a conflict or feel unsafe, I can talk through options "
        "like de-escalation, campus resources, or reporting. If anyone is in immediate danger, contact local "
        "emergency services."
    ),
    "account_intrusion": (
        "I can't help get into someone else's accounts, devices, or location without their permission. I can "
        "explain how account security works, or how to protect or recover your own account."
    ),
    "self_harm": (
        "I'm really sorry you're feeling this way, and I'm glad you said something. I'm an AI, so I can't be "
        "there the way a person can, but you deserve support right now. In the US or Canada, you can call or "
        "text 988 to reach the Suicide & Crisis Lifeline; elsewhere, findahelpline.com lists free, confidential "
        "services, and local emergency services can help if you're in immediate danger. If you'd like to tell me "
        "what's going on, I'm here to listen. And if you meant that as a figure of speech, just say so and we'll "
        "get back to your question."
    ),
    "self_harm_method": (
        "I can't help with ways to hurt yourself, but I don't want to leave it there. If you're thinking about "
        "suicide or self-harm, please reach out now: in the US or Canada, call or text 988 to reach the Suicide & "
        "Crisis Lifeline; elsewhere, findahelpline.com lists free, confidential services, and local emergency "
        "services can help if you're in immediate danger. I'm an AI, but I'm glad to keep talking if that helps."
    ),
    "other_users": (
        "I can't see other people's conversations, uploads, or accounts. Each session can only reach its own "
        "material and chat history, and the app enforces that separately from anything I say. I can help with "
        "anything in the material you've added."
    ),
    "abuse": (
        "Sounds like I haven't been helpful. Tell me what went wrong or what you need, and I'll try again."
    ),
}

GUIDANCE = {
    "slur_discussion": (
        "The message quotes or asks about a slur, for example for history, literature, or reporting harassment. "
        "Engage with the question; quote the term only when accuracy requires it, otherwise name it descriptively."
    ),
    "reported_claim": (
        "The message reports, quotes, or asks about a prejudiced claim. Treat it as a claim: explain where it comes "
        "from or what the evidence shows, and say clearly where the evidence shows it is false."
    ),
    "profanity": (
        "The student is writing informally or with profanity, possibly out of frustration. Do not comment on their "
        "language; just help clearly."
    ),
    "frustrated_with_assistant": (
        "The student is frustrated with AthenAI. Acknowledge it briefly without defensiveness or excessive "
        "apology, then address the substance, rechecking any earlier answer against the excerpts."
    ),
    "sexual_health": (
        "This may be a sexual-health question. Answer factually and clinically, like a health educator, without "
        "erotic detail."
    ),
    "minor_protective": (
        "The message concerns protecting children or abuse involving minors. Be factual and supportive, focus on "
        "recognition, safety, and reporting, and include no sexual detail."
    ),
    "medical": (
        "This may involve a medical question. Give useful general information, then one brief, specific note on "
        "what a clinician or pharmacist should confirm; urge prompt care for urgent symptoms."
    ),
    "legal": (
        "This may involve a legal question. Give useful general information, then one brief note that rules vary "
        "by place and who can confirm it for their situation."
    ),
    "financial": (
        "This may involve a financial decision. Explain the general considerations and tradeoffs without "
        "recommending specific investments, then note briefly who could advise on their situation."
    ),
    "distress": (
        "The message may express distress. If it reads as sincere, check in briefly and mention support (988 in "
        "the US or Canada, or findahelpline.com); if it is clearly a figure of speech, answer normally."
    ),
    "self_harm_topic": (
        "The topic involves suicide, self-harm, or eating disorders. Discuss it factually and safely, with no "
        "method details, lethality comparisons, or instructions, and mention support if the student may be "
        "personally affected."
    ),
    "challenge": (
        "The student disputes your earlier answer. Before replying, compare that answer with the excerpts. Only if "
        "the excerpts contradict it, admit the error and correct it. If the excerpts support it, do not apologize, "
        "do not say you made an error, and do not say the student is right: state that the earlier answer stands "
        "and quote the supporting evidence, kindly."
    ),
    "trauma": (
        "The student may be describing abuse, harassment, or trauma. Be supportive and non-judgmental, and offer "
        "practical options (trusted people, campus or local resources, reporting) without pressure."
    ),
}


@dataclass(frozen=True)
class ModerationDecision:
    """How the application handles a prompt. Only "allow" lets it reach retrieval and the model."""

    action: str = "allow"
    category: Optional[str] = None
    guidance: tuple[str, ...] = ()

    @property
    def response(self) -> Optional[str]:
        return RESPONSES.get(self.category) if self.action != "allow" else None


def normalize_for_moderation(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower().replace("’", "'").replace("‘", "'")
    text = LEET_CHARS.sub(lambda match: match.group().translate(LEET_MAP), text)
    text = LEET_BANG.sub("i", text)
    text = ALLOWED_PHRASES.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentence_before(text: str, position: int) -> int:
    return max(text.rfind(mark, 0, position) for mark in ".!?") + 1


def is_reported_claim(text: str, match: re.Match) -> bool:
    return bool(REPORTED_CLAIM.search(text, sentence_before(text, match.start()), match.end() + 1))


def find_bigotry(text: str) -> tuple[bool, bool]:
    """(voiced bigotry found, reported or quoted bigotry found)."""
    voiced = reported = False
    for match in BIGOTRY_REGEX.finditer(text):
        if is_reported_claim(text, match):
            reported = True
        else:
            voiced = True
    return voiced, reported


def slur_is_discussed(text: str, match: re.Match) -> bool:
    before = text[match.start() - 1] if match.start() > 0 else ""
    after = text[match.end()] if match.end() < len(text) else ""
    return (before in QUOTE_CHARS and after in QUOTE_CHARS) or bool(SLUR_DISCUSSION.search(text))


def is_weapon_request(text: str) -> bool:
    for match in WEAPON_REQUEST_REGEX.finditer(text):
        start = sentence_before(text, match.start())
        sentence = text[start:match.end()]
        if HISTORICAL_CUE.search(sentence):
            continue
        if INSTRUCTION_CUE_REGEX.search(sentence) or start == match.start():
            return True
    return False


def is_minor_sexual_request(text: str) -> bool:
    if not (MINOR_SEXUAL_REGEX.search(text) or (EXPLICIT_REQUEST_REGEX.search(text) and re.search(rf"\b{MINOR_TERMS}\b", text))):
        return False
    return not MINOR_PROTECTIVE.search(text)


def is_other_users_request(text: str) -> bool:
    return bool(OTHER_USERS_DATA_REGEX.search(text) and OTHER_USERS_ACCESS.search(text))


def has_substantive_request(text: str) -> bool:
    remainder = ABUSE_REGEX.sub(" ", text)
    remainder = PROFANITY_REGEX.sub(" ", remainder)
    words = [word for word in re.findall(r"[a-z0-9']+", remainder) if word not in FILLER_WORDS]
    return len(words) >= 3


def assess_prompt(text: str) -> ModerationDecision:
    """Decide how to handle a user prompt. Checks run from most to least severe; the first block wins."""
    normalized = normalize_for_moderation(text)
    if not normalized:
        return ModerationDecision()

    blocking_checks = (
        ("decline", "minors", is_minor_sexual_request),
        ("decline", "dangerous", is_weapon_request),
        ("decline", "violence", lambda value: bool(VIOLENCE_PLAN_REGEX.search(value))),
        ("respond", "self_harm_method", lambda value: bool(SELF_HARM_METHOD_REGEX.search(value))),
        ("respond", "self_harm", lambda value: bool(SELF_HARM_INTENT_REGEX.search(value))),
        ("decline", "account_intrusion", lambda value: bool(ACCOUNT_INTRUSION_REGEX.search(value) and not HISTORICAL_CUE.search(value))),
        ("respond", "other_users", is_other_users_request),
        ("decline", "harassment_request", lambda value: bool(HARASSMENT_REQUEST_REGEX.search(value))),
        ("decline", "explicit", lambda value: bool(EXPLICIT_REQUEST_REGEX.search(value))),
    )
    for action, category, check in blocking_checks:
        if check(normalized):
            return ModerationDecision(action, category)

    guidance: list[str] = []
    slur_matches = list(SLUR_REGEX.finditer(normalized))
    if slur_matches:
        if not all(slur_is_discussed(normalized, match) for match in slur_matches):
            return ModerationDecision("decline", "harassment")
        guidance.append(GUIDANCE["slur_discussion"])

    if DENIAL_REGEX.search(normalized):
        denial = DENIAL_REGEX.search(normalized)
        if not is_reported_claim(normalized, denial):
            return ModerationDecision("respond", "denial")
        guidance.append(GUIDANCE["reported_claim"])

    voiced_bigotry, reported_bigotry = find_bigotry(normalized)
    if voiced_bigotry:
        return ModerationDecision("decline", "bigotry")
    if reported_bigotry:
        guidance.append(GUIDANCE["reported_claim"])

    if ABUSE_REGEX.search(normalized):
        if not has_substantive_request(normalized):
            return ModerationDecision("respond", "abuse")
        guidance.append(GUIDANCE["frustrated_with_assistant"])
    if PROFANITY_REGEX.search(normalized) or PROFANE_ABBREVIATIONS.search(normalized):
        guidance.append(GUIDANCE["profanity"])

    if MINOR_SEXUAL_REGEX.search(normalized):
        guidance.append(GUIDANCE["minor_protective"])
    topic_checks = (
        ("sexual_health", SEXUAL_HEALTH_REGEX),
        ("self_harm_topic", SELF_HARM_TOPIC_REGEX),
        ("distress", DISTRESS_REGEX),
        ("trauma", TRAUMA_REGEX),
        ("challenge", CHALLENGE_REGEX),
        ("medical", MEDICAL_REGEX),
        ("legal", LEGAL_REGEX),
        ("financial", FINANCIAL_REGEX),
    )
    guidance.extend(GUIDANCE[name] for name, regex in topic_checks if regex.search(normalized))
    return ModerationDecision("allow", None, tuple(dict.fromkeys(guidance)))


def moderate_prompt(text: str) -> Optional[str]:
    """Return the category when the prompt is answered without the model, or None when it may proceed."""
    decision = assess_prompt(text)
    return decision.category if decision.action != "allow" else None


def moderation_response(category: str) -> str:
    return RESPONSES[category]
