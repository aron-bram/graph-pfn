import numpy as np
import pytest

from utils import (ArgSampler, DummySampler, PriorArgList,
                             PriorArgRange)


@pytest.fixture
def separators():
    return {"sep": ",", "r_sep": "--", "w_sep": "__"}


def test_sampler_is_priorargrange_when_arg_is_two_ints_separated_by_rsep(
    separators: dict,
):
    nfeat = ArgSampler(f"10{separators['r_sep']}100", "nfeat")
    assert isinstance(nfeat.sampler, PriorArgRange)


def test_sampler_is_priorarglist_when_arg_is_list_of_ints(separators: dict):
    sep = separators["sep"]
    nnode = ArgSampler(f"10{sep}100{sep}1000", "nnode")
    assert isinstance(nnode.sampler, PriorArgList)


def test_sampler_samples_ints_when_ints_are_given(separators: dict):
    sep = separators["sep"]
    nnode = ArgSampler(f"10{sep}100{sep}1000", "nnode")
    assert all([isinstance(v, int) for v in nnode.sampler.value])
    samples = nnode.sample(n=10)
    assert all([isinstance(s, int) for s in samples.tolist()])


def test_sampler_samples_floats_when_floats_are_given(separators: dict):
    sep = separators["sep"]
    nnode = ArgSampler(f"1.0{sep}2{sep}3", "nnode")
    assert all([isinstance(v, float) for v in nnode.sampler.value])
    samples = nnode.sample(n=10)
    assert all([isinstance(s, float) for s in samples])


def test_sampler_is_dummysampler_when_arg_is_either_int_float_bool_or_str_without_sep(
    separators: dict,
):
    nnode = ArgSampler("1000", "nnode")
    dropout = ArgSampler("0.1", "dropout")
    p = ArgSampler(0.1, "p")
    nfeat = ArgSampler(111, "nfeat")
    bias = ArgSampler("True", "bias")
    do = ArgSampler(False, "do")
    s = ArgSampler("abc", "s")
    str_with_sep = ArgSampler(f"abc{separators['sep']}", "str_with_sep")

    assert isinstance(nnode.sampler, DummySampler)
    assert isinstance(dropout.sampler, DummySampler)
    assert isinstance(p.sampler, DummySampler)
    assert isinstance(nfeat.sampler, DummySampler)
    assert isinstance(bias.sampler, DummySampler)
    assert isinstance(do.sampler, DummySampler)
    assert isinstance(s.sampler, DummySampler)
    assert not isinstance(str_with_sep.sampler, DummySampler)


def test_sampler_samples_int_float_str_bool_when_arg_is_dummysampler():
    nnode = ArgSampler("1000", "nnode").sample()
    dropout = ArgSampler("0.1", "dropout").sample()
    p = ArgSampler(0.1, "p").sample()
    nfeat = ArgSampler(111, "nfeat").sample()
    bias = ArgSampler("True", "bias").sample()
    do = ArgSampler(False, "do").sample()
    s = ArgSampler("abc", "s").sample()

    assert nnode.item() == 1000
    assert dropout.item() == 0.1
    assert p.item() == 0.1
    assert nfeat.item() == 111
    assert bias.item() == True
    assert do.item() == False
    assert s.item() == "abc"


def test_sampler_converts_input_str_to_list_without_duplicates(separators: dict):
    sep = separators["sep"]
    arg = ArgSampler(f"c{sep}a{sep}b_b{sep}c{sep}a{sep}c{sep}c", "arg")
    assert isinstance(arg.sampler, PriorArgList)
    value = ["a", "b_b", "c"]
    assert sorted(arg.value) == value


def test_sampler_samples_from_list_when_it_is_a_priorarglist(separators: dict):
    sep = separators["sep"]
    arg = ArgSampler(f"a{sep}b_b{sep}c", "arg")
    assert isinstance(arg.sampler, PriorArgList)
    samplable = ["a", "b_b", "c"]
    samples = [arg.sample() for i in range(10)]
    assert all([sample in samplable for sample in samples])


def test_sampler_weighs_elements_when_weight_is_passed(separators: dict):
    sep = separators["sep"]
    w_sep = separators["w_sep"]

    arg = ArgSampler(f"a{sep}b{w_sep}1{sep}0", "arg")
    samples = [arg.sample() for i in range(10)]
    assert "b" not in samples and samples.count("a") == len(samples)


def test_sampler_samples_floats_from_range_when_range_has_a_float(separators: dict):
    r_sep = separators["r_sep"]
    arg1 = ArgSampler(f"0.1{r_sep}5", "arg1")
    arg2 = ArgSampler(f"0{r_sep}12", "arg2")
    arg3 = ArgSampler(f"0{r_sep}1", "arg3")
    assert all([isinstance(v, float) for v in arg1.value])
    assert all([not isinstance(v, float) for v in arg2.value])
    assert all([isinstance(v, float) for v in arg3.value])
    samples1 = [arg1.sample() for i in range(10)]
    samples2 = [arg2.sample() for i in range(10)]
    samples3 = [arg3.sample() for i in range(10)]
    assert all([isinstance(s, float) for s in samples1])
    assert all([not isinstance(s, float) for s in samples2])
    assert all([isinstance(s, float) for s in samples3])


def test_sampler_samples_ints_from_range_when_range_has_only_ints(separators: dict):
    r_sep = separators["r_sep"]
    arg1 = ArgSampler(f"-4{r_sep}5", "arg1")
    arg2 = ArgSampler(f"0{r_sep}1.2", "arg2")
    assert all([isinstance(v, int) for v in arg1.value])
    assert not all([isinstance(v, int) for v in arg2.value])
    samples1 = arg1.sample(n=10)
    samples2 = arg2.sample(n=10)
    assert all([isinstance(s.item(), int) for s in samples1])
    assert all([not isinstance(s.item(), int) for s in samples2])


def test_sampler_samples_constant_when_range_has_identical_min_and_max(
    separators: dict,
):
    arg = ArgSampler("2--2", "arg")
    assert all([arg.sample(n=1) == 2 for i in range(10)])
    arg = ArgSampler("2.1--2.1", "arg")
    assert all([arg.sample(n=1) == 2.1 for i in range(10)])


def test_sampler_samples_constant_when_one_unique_value_repeats(separators: dict):
    arg1 = ArgSampler("a,a,a", "arg")
    assert all([arg1.sample(n=1) == "a" for i in range(10)])
    arg2 = ArgSampler("1,1", "arg")
    assert all([arg2.sample(n=1) == 1 for i in range(10)])
