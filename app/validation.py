# app/validation.py
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CreditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=100)
    revenu_mensuel: float = Field(gt=0, le=10_000_000)
    montant_credit: float = Field(gt=0, le=5_000_000)
    duree_mois: int = Field(ge=1, le=360)
    anciennete_activite: float = Field(ge=0, le=60)

    @field_validator(
        "age",
        "revenu_mensuel",
        "montant_credit",
        "duree_mois",
        "anciennete_activite",
        mode="before",
    )
    @classmethod
    def _reject_non_scalar_numeric_values(cls, v, info):
        if v is None:
            return v

        if isinstance(v, bool):
            raise ValueError(f"{info.field_name} doit être numérique")

        if isinstance(v, (int, float)):
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError("Valeur non finie interdite (NaN, inf, -inf)")
            return v

        if isinstance(v, str):
            stripped = v.strip()
            if stripped == "":
                raise ValueError(f"{info.field_name} doit être numérique")
            lower = stripped.lower()
            if lower in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "null", "none", "true", "false"}:
                raise ValueError("Valeur non finie interdite (NaN, inf, -inf)")
            try:
                numeric = float(stripped)
            except ValueError as exc:
                raise ValueError(f"{info.field_name} doit être numérique") from exc
            if not math.isfinite(numeric):
                raise ValueError("Valeur non finie interdite (NaN, inf, -inf)")
            return numeric

        raise ValueError(f"{info.field_name} doit être numérique")

    @model_validator(mode="after")
    def coherence_anciennete_age(self):
        if self.anciennete_activite > (self.age - 18):
            raise ValueError("anciennete_activite incohérente avec l'âge")
        return self