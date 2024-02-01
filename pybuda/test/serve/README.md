# Bert QA

This is a simple demo of Bert QA using PyBuda.

## Env setup

Install Ray Serve library:

```
pip install "ray[serve]"
```

And, start it:

```
ray start --head
```

When done, to stop it, run:

```
ray stop
```

## Deploying QA Serve

```
python qa_serve.py
```

This will create and deploy a server that will serve up answers to QA questions.


## Ask questions

```
python ask.py
```

Enter a question for the given context, and the request will be sent to the server. The first request made will take a while because
the model will be compiled and installed on the device. Subsequent requests should be very fast. Feel free to run `ask.py` repeatedly.


