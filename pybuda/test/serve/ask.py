# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import requests
import json
#context = "Manuel Romero has been working hardly in the repository hugginface/transformers lately"
#input_q = {"context": context, "question": "For which company has worked Manuel Romero?"}
context = "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."

while True:
    print("Context: ")
    print(context)
    print("Question ('quit' to quit): ")
    question = input()
    if question == "quit":
        exit(0)

    input_q = {"context": context, "question": question}
    print("Running your question through Bert on Tenstorrent Hardware")
    result = requests.get("http://127.0.0.1:8000/bert-qa", data=json.dumps(input_q)).text
    result = json.loads(result)

    print("****************")
    print("Question: ", question)
    print("Answer: ", result["answer"])
    print("\nDetails: ", result)
    print("****************")

