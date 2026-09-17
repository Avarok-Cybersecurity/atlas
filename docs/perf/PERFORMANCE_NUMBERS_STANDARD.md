# Standard for Performance Numbers

The diligence page at [atlasinference.io/diligence](https://atlasinference.io/diligence) is the single public record, and our internal best numbers and the page must agree.

## Official guidelines

1. Any internal result that beats the page goes onto the page by PR, with the following:
   a. launch line: every parameter and switch, model and drafter locations
   b. commit sha
   c. driver line: the benchmark command and its settings
   d. receipt JSON

2. Any page number an internal run cannot reproduce within its published spread:
   a. gets a note in the channel with the run's receipt
   b. gets the page corrected by PR

3. Every table posted:
   a. same driver, same settings
   b. medians over reps
   c. competing engines in the same table
   d. TTFT p50 alongside tok/s, per rung

A number without a launch line is a lab note, not a claim.
