import evaluate

generated_summary = ['The fox jumped over the dog.']
reference_summary = ['The quick brown fox jumped over the lazy grey dog.']

rouge = evaluate.load("rouge")
results = rouge.compute(
    predictions=generated_summary,
    references=reference_summary
)
print(results)


results = rouge.compute(
    predictions=['The child ran for the pickle ball.'],
    references=generated_summary
)
print(results)