"""Shared selection contract: preserve personal tax evidence, omit generic prose."""
USEFUL_CONTENT = """Keep only these categories of source evidence:
1. Populated tax-form fields: form type/year, box/line identifiers and their
   labels, exact amounts, dates, codes, checked elections, and correction flags.
2. Actual financial records: income, distributions, interest, dividends, gains,
   losses, proceeds, basis, adjustments, withholding, contributions, deductible
   expense candidates, loan interest/principal, donations and receipts. Do not
   infer taxability, deductibility or eligibility from the presence of a value.
3. Actual transaction rows and their tax-relevant columns (security/item,
   acquisition/sale/payment dates, amount, proceeds, basis, adjustments, covered
   status, term and withholding). Retain every relevant row; do not replace
   transactions with totals alone. Keep supplied subtotals/totals separately.
4. Minimal context linking records: taxpayer/recipient and payer names, account
   identifiers, statement period, tax year, section/table headers and units.
5. A note ONLY when it specifically explains this taxpayer's reported record,
   correction, missing basis, adjustment, withholding, ownership or election.
   Retain the relevant factual statement or exact linked code/footnote text.
Omit generic filing instructions, box-definition essays, tax advice, examples,
standard legal disclosures, privacy policies, contact directories, repeated
mailing headers, marketing, and unrelated narrative. A number inside generic
instructions or an example is NOT a personal financial record. A footnote's
mere mention of taxes or basis does not make generic guidance relevant.
Preserve linked exceptions and annotations needed to interpret actual records.
If relevance is uncertain, retain the potentially relevant content for review;
never decide based on a guessed future query or silently drop transaction rows.
"""
