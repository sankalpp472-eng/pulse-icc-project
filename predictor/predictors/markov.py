"""markov: order-2 Markov chain over actions, P(next | last two tools), backing off to the last tool and then
to the overall action frequencies (action_model.NGramModel). Reads the trajectory only."""

from predictors.action import ActionPickle


class Markov(ActionPickle):
    name = "markov"
    key = "ngram"
    description = "Order-2 Markov chain over actions with back-off: P(next | last two tools)"
    inputs = ("trajectory",)
    supports_fanout = False        # the pending-ID rule needs signals, which markov does not read
