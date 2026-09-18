from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


def _ascii_fold(s: str) -> str:
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append(ch)
        else:
            nfkd = unicodedata.normalize("NFKD", ch)
            for c in nfkd:
                if not unicodedata.combining(c) and ord(c) < 128:
                    out.append(c)
                    break
            else:
                out.append("-")
    return "".join(out)


_SLUG_COLLAPSE_RE = re.compile(r"[^a-z0-9]+")


def safe_slug(title: str, max_len: int = 80) -> str:
    t = unicodedata.normalize("NFC", title).strip()
    folded = _ascii_fold(t).lower()
    slug = _SLUG_COLLAPSE_RE.sub("-", folded).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "page"


@dataclass(frozen=True)
class FieldSpec:
    canonical: str
    aliases: tuple[str, ...] = ()
    multi: bool = False

    def matches(self, key: str) -> bool:
        k = key.strip().lower().replace(" ", "_")
        if k == self.canonical.lower():
            return True
        return k in {a.lower() for a in self.aliases}


@dataclass(frozen=True)
class TemplateSpec:
    template_name: str
    entity_type: str
    fields: tuple[FieldSpec, ...]
    infobox_marker: str | None = None


@dataclass
class Registry:
    templates: dict[str, TemplateSpec] = field(default_factory=dict)
    category_to_entity: dict[str, str] = field(default_factory=dict)
    dir_to_entity: dict[str, str] = field(default_factory=dict)
    denylist_templates: set[str] = field(default_factory=set)
    navbox_templates: set[str] = field(default_factory=set)
    infobox_prefixes: tuple[str, ...] = ("Infobox", "DaSI", "DaSII", "DaSIII")

    @classmethod
    def default(cls) -> Registry:
        r = cls()
        r._register_defaults()
        return r

    def _register_defaults(self) -> None:
        common_dsiii = (
            "name",
            "image",
            "weapon-type",
            "atk-type",
            "psn-atk",
            "bld-atk",
            "fst-atk",
            "spc-atk",
            "focus-req",
            "ring-cost",
            "cast-bonus",
            "shot-range",
            "stability",
            "durability",
            "weight",
            "counter",
            "phys-atk",
            "mag-atk",
            "fire-atk",
            "ltn-atk",
            "dark-atk",
            "phys-def",
            "mag-def",
            "fire-def",
            "ltn-def",
            "dark-def",
            "str-req",
            "dex-req",
            "int-req",
            "fth-req",
            "str-bonus",
            "dex-bonus",
            "int-bonus",
            "fth-bonus",
            "buy-price",
            "price",
            "notes",
        )
        weapon_t = TemplateSpec(
            template_name="DaSIIIWeapon",
            entity_type="weapon",
            fields=tuple(FieldSpec(name) for name in common_dsiii),
            infobox_marker="DaSIIIWeapon",
        )
        self.templates[weapon_t.template_name] = weapon_t

        char_fields = (
            FieldSpec("theme"),
            FieldSpec("image"),
            FieldSpec("appears_in", multi=True),
            FieldSpec("mentioned_in", multi=True),
            FieldSpec("located_in", multi=True),
            FieldSpec("voiced_by"),
            FieldSpec("drops"),
            FieldSpec("title"),
        )
        char_t = TemplateSpec(
            template_name="Infobox character",
            entity_type="character",
            fields=char_fields,
            infobox_marker="Infobox character",
        )
        self.templates[char_t.template_name] = char_t

        self.category_to_entity.update(
            {
                "Weapons": "weapon",
                "Armor": "armor",
                "Shields": "shield",
                "Rings": "ring",
                "Items": "item",
                "Characters": "character",
                "Locations": "location",
                "Enemies": "enemy",
                "Bosses": "boss",
                "Spells": "spell",
                "Miracles": "spell",
                "Pyromancies": "spell",
                "Sorceries": "spell",
                "Mechanics": "mechanic",
                "Status Effects": "mechanic",
            }
        )

        self.dir_to_entity.update(
            {
                "Weapons": "weapon",
                "Armor": "armor",
                "Shields": "shield",
                "Rings": "ring",
                "Items": "item",
                "Characters": "character",
                "NPCs": "character",
                "Mentioned Characters": "character",
                "Blacksmiths": "character",
                "Bosses": "boss",
                "Locations": "location",
                "Mentioned Locations": "location",
                "Enemies": "enemy",
                "Magic": "spell",
                "Gameplay Mechanics": "mechanic",
            }
        )

        for name in (
            "clr",
            "Main",
            "For",
            "Note",
            "Japanese",
            "Quoteleft",
            "Quote",
            "Disambig",
            "stub",
            "lang",
            "Missing",
            "stub",
            "!",
            "DaSIIIMoveset",
            "DaSIIISWB",
            "DaSIIIWeaponBox",
            "DaSIIIWeaponsNav",
        ):
            self.denylist_templates.add(name.lower())
        for p in ("DaSIIIWeaponTypeBox", "DaSIIWeaponTypeBox", "DaSIWeaponTypeBox"):
            self.navbox_templates.add(p.lower())

    def is_denylist_template(self, name: str) -> bool:
        n = name.strip().lower()
        if n in self.denylist_templates:
            return True
        if n in self.navbox_templates:
            return True
        for p in self.navbox_templates:
            if n.startswith(p.lower() + "/"):
                return True
        return bool(n.endswith("nav") and not n.startswith("infobox"))

    def find_infobox_spec(self, template_name: str) -> TemplateSpec | None:
        return self.templates.get(template_name.strip())

    def entity_from_category(self, category: str) -> str | None:
        parts = category.split(":", 1)
        last = parts[-1].strip()
        for k, v in self.category_to_entity.items():
            if (
                last == k
                or last.lower() == k.lower()
                or last.endswith((f": {k}", f":{k}"))
            ):
                return v
        return None

    def entity_from_directory(self, segment: str) -> str | None:
        return self.dir_to_entity.get(segment)
