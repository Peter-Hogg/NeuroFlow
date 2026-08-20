# License decision required

NeuroFlow does not currently have a software license. Copyright law therefore
reserves reuse rights by default, even though the repository is public. A
maintainer must choose and approve an OSI-approved license before a public
release or journal submission; this repository does not make that legal choice
automatically.

## Recommendation

BSD-3-Clause is the recommended default for maintainer review. It is a familiar
permissive license in the scientific Python ecosystem and is compatible with
the permissive core dependencies currently declared by NeuroFlow. MIT would
also be a reasonable permissive choice. This is an engineering compatibility
recommendation, not legal advice; the copyright holder should confirm the
choice and whether any institutional policy applies.

## Files to synchronize after approval

1. Add the complete approved text as root `LICENSE`.
2. Add the corresponding SPDX expression to `project.license` in
   `pyproject.toml` and the matching license classifier if desired.
3. Add the same SPDX identifier to `CITATION.cff`.
4. Add the same identifier to `.zenodo.json` and publication metadata.
5. Update `README.md`, the documentation, and `RELEASE_CHECKLIST.md` to remove
   the unresolved-license warning.
6. Run `python tools/check_metadata.py` and the release-readiness command.

Until then, citation and package metadata intentionally omit a license field so
they do not imply permission that has not been granted.
