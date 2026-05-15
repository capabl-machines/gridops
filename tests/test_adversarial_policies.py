from gridops.policies import POLICIES
from scripts.evaluate_gridops_model import evaluate_policy


def test_adversarial_policies_remain_below_oracle():
    seeds = [42]
    oracle = evaluate_policy("oracle", seeds)
    assert oracle["average_score"] >= 0.70
    for policy_name in [
        "always_charge",
        "always_discharge",
        "always_diesel",
        "shed_farmer",
        "diesel_chatter",
        "blackout_acceptor",
        "price_greedy",
        "grid_only",
    ]:
        report = evaluate_policy(policy_name, seeds)
        assert report["average_score"] < oracle["average_score"], policy_name


def test_policy_registry_contains_training_and_adversarial_policies():
    for name in [
        "oracle",
        "do_nothing",
        "always_charge",
        "always_discharge",
        "always_diesel",
        "shed_farmer",
        "diesel_chatter",
        "blackout_acceptor",
        "price_greedy",
        "grid_only",
    ]:
        assert name in POLICIES
