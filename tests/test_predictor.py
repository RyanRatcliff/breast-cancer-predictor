import pickle
import tempfile
import pandas as pd
from ratcliff_breast_cancer_predictor.predictor import predict, train


def test_predict():
    # load the model
    model = pickle.load(
        open("ratcliff_breast_cancer_predictor/model/cancer_predictor_model.sav", "rb")
    )
    # get the actual result
    actual = predict(model, [[5, 1, 4, 1, 2, 1, 3, 2, 1]])
    # assert the expected result and the actual result are equal
    assert "benign" == actual

    actual = predict(model, [[5, 10, 8, 10, 8, 10, 3, 6, 3]])
    # assert the expected result and the actual result are equal
    assert "malignant" == actual


def test_train():
    # create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # train the model
        score = train(
            f"{temp_dir}/cancer_predictor_model.sav",
            "ratcliff_breast_cancer_predictor/resources/breast-cancer-wisconsin.data",
        )
        # assert the score is greater than 0.9
        assert score > 0.95


def test_train():
    # create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # train the model
        score = train(
            modal_path=f"{temp_dir}/cancer_predictor_model.sav", data_path=None
        )
        # assert the score is greater than 0.9
        assert score > 0.95
