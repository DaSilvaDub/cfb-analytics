"""Exception hierarchy.

Every failure that a caller might reasonably want to distinguish gets its own
type. Nothing in this package raises a bare ``Exception``, and nothing swallows
an error to return a default: a missing source is reported, not imputed.
"""

from __future__ import annotations


class CfbAnalyticsError(RuntimeError):
    """Base class for every error raised by this package."""


class ConfigError(CfbAnalyticsError):
    """Configuration is missing or malformed."""


class MissingCredentialError(ConfigError):
    """A required credential is not configured.

    Carries the exact environment-variable name so the message can tell the
    operator what to set rather than making them read source.
    """

    def __init__(self, env_var: str, purpose: str, how_to_obtain: str = "") -> None:
        self.env_var = env_var
        self.purpose = purpose
        message = f"Missing credential: set the {env_var} environment variable ({purpose})."
        if how_to_obtain:
            message += f" {how_to_obtain}"
        super().__init__(message)


class SourceError(CfbAnalyticsError):
    """A data source failed."""


class AuthRequiredError(SourceError):
    """The source rejected the request; the saved session needs refreshing."""


class UnknownLeagueError(SourceError):
    """The source does not recognise the requested league token."""


class SchemaError(CfbAnalyticsError):
    """A payload did not have the shape the ingester requires."""


class LeakageError(CfbAnalyticsError):
    """A feature read would have used information unavailable before kickoff.

    Raised by ``features.asof.AsOfReader``. This is deliberately fatal: a
    backtest that leaks looks excellent and is worthless, so the failure mode
    must be a crash rather than an optimistic number.
    """
