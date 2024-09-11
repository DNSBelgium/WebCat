import builtins
import pathlib
from unittest.mock import mock_open

import fastapi
import pandas
import pytest

import webserver as w

pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(autouse=True)
def run_before_each_test():
    w.PREPARE_DICT = {}
    w.PREDICTION_DICT = {}


@pytest.mark.asyncio
async def test_prepare_status_job_done():
    w.PREPARE_DICT['job1'] = w.Condition(
        err=None,
        work_in_progress=False
    )
    response = w.Response()

    result = await w.get_status_prepare("job1", response)

    assert result.job_id == "job1"
    assert result.message == "job done"
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_prepare_status_job_in_progress():
    w.PREPARE_DICT['job1'] = w.Condition(
        err=None,
        work_in_progress=True
    )
    response = w.Response()

    result = await w.get_status_prepare("job1", response)

    assert result.job_id == "job1"
    assert result.message == "job in progress"
    assert response.status_code == 423


@pytest.mark.asyncio
async def test_prepare_status_job_not_found():
    response = w.Response()
    result = await w.get_status_prepare("job1", response)
    assert result.message == "job not found"
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_prepare_status_job_error():
    w.PREPARE_DICT['job1'] = w.Condition(
        err=Exception("test"),
        work_in_progress=False
    )

    response = w.Response()
    result = await w.get_status_prepare("job1", response)
    assert result.message.startswith("something went wrong:")
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_prepare_parquet_happy(mocker):
    res = w.Response()

    mock_background_task = mocker.patch.object(fastapi.BackgroundTasks, "add_task")
    mock_UploadFile = mocker.patch.object(fastapi.UploadFile, "read")
    mock_open = mocker.patch.object(builtins, "open")
    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = True

    resp_json = await w.prepare_parquet("test.model", mock_UploadFile, mock_background_task, res)

    assert res.status_code == 201
    assert resp_json.message == "prepare started"
    assert len(resp_json.job_id) > 0

    mock_background_task.add_task.assert_called_once()
    mock_open.assert_called_once()


@pytest.mark.asyncio
async def test_prepare_parquet_model_does_not_exist(mocker):
    res = w.Response()

    mock_background_task = mocker.patch.object(fastapi.BackgroundTasks, "add_task")
    mock_UploadFile = mocker.patch.object(fastapi.UploadFile, "read")

    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = False

    resp_json = await w.prepare_parquet("test.model", mock_UploadFile, mock_background_task, res)

    assert res.status_code == 400
    assert resp_json.message == "model not found"
    assert resp_json.job_id == ""


@pytest.mark.asyncio
async def test_prepare_parquet_task_already_running(mocker):
    res = w.Response()
    w.PREPARE_DICT = {'id1': w.Condition(None, True)}

    mock_background_task = mocker.patch.object(fastapi.BackgroundTasks, "add_task")
    mock_UploadFile = mocker.patch.object(fastapi.UploadFile, "read")
    mocker.patch.object(builtins, "open")

    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = True

    resp_json = await w.prepare_parquet("test.model", mock_UploadFile, mock_background_task, res)

    assert res.status_code == 423
    assert resp_json.message == "preparation in progress"


@pytest.mark.asyncio
async def test_prepare_parquet_data_input_wrong(mocker):
    res = w.Response()

    mock_background_task = mocker.patch.object(fastapi.BackgroundTasks, "add_task")
    mocker.patch.object(builtins, "open")
    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = True

    resp_json = await w.prepare_parquet("test.model", None, mock_background_task, res)

    assert res.status_code == 500
    assert resp_json.message == "there was an error uploading the parquet file"


@pytest.mark.asyncio
async def test_prepare_parquet_data_could_not_write(mocker):
    res = w.Response()

    mock_background_task = mocker.patch.object(fastapi.BackgroundTasks, "add_task")
    mock_UploadFile = mocker.patch.object(fastapi.UploadFile, "read")
    mock_open = mocker.patch.object(builtins, "open")
    mock_open.side_effect = Exception()
    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = True

    resp_json = await w.prepare_parquet("test.model", mock_UploadFile, mock_background_task, res)

    assert res.status_code == 500
    assert resp_json.message == "there was an error uploading the parquet file"


@pytest.mark.asyncio
async def test_prepare_json_happy(mocker):
    prepare = w.PrepareJson(
        domain_name=["test1.de", "test2.de"],
        body_text=["testdata for test1", "testdata for test2"],
        meta_text=["meta1", "meta2"],
        model='jeder zottel ist ein model')

    res = w.Response()

    mock_background_task = mocker.patch.object(fastapi.BackgroundTasks, "add_task")
    mock_write_preparation_parquet_from_body = mocker.patch.object(w, "write_preparation_parquet_from_body")
    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = True

    resp_json = await w.prepare_json(prepare, mock_background_task, res)

    assert res.status_code == 201
    assert resp_json.message == "prepare started"
    assert len(resp_json.job_id) > 0

    mock_background_task.add_task.assert_called_once()
    mock_write_preparation_parquet_from_body.assert_called_once()


@pytest.mark.asyncio
async def test_prepare_json_model_not_exist(mocker):
    prepare = w.PrepareJson(
        domain_name=["test1.de", "test2.de"],
        body_text=["testdata for test1", "testdata for test2"],
        meta_text=["meta1", "meta2"],
        model='jeder zottel ist ein model')

    res = w.Response()

    mock_background_task = mocker.patch.object(fastapi.BackgroundTasks, "add_task")
    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = False

    resp_json = await w.prepare_json(prepare, mock_background_task, res)

    assert res.status_code == 400
    assert resp_json.message == "model not found"
    assert resp_json.job_id == ""


@pytest.mark.asyncio
async def test_prepare_json_data_input_wrong(mocker):
    prepare = w.PrepareJson(
        domain_name=["test1.de", "test2.de"],
        body_text=["testdata for test1", "testdata for test2"],
        meta_text=["meta1", "meta2"],
        model='jeder zottel ist ein model')

    res = w.Response()

    mock_background_task = mocker.patch.object(fastapi.BackgroundTasks, "add_task")
    mock_write_preparation_parquet_from_body = mocker.patch.object(w, "write_preparation_parquet_from_body")
    mock_write_preparation_parquet_from_body.side_effect = ValueError('test exception')
    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = True

    resp_json = await w.prepare_json(prepare, mock_background_task, res)

    assert res.status_code == 406
    assert resp_json.message == "input data was not correct: test exception"


@pytest.mark.asyncio
async def test_prepare_json_task_already_running(mocker):
    prepare = w.PrepareJson(
        domain_name=["test1.de", "test2.de"],
        body_text=["testdata for test1", "testdata for test2"],
        meta_text=["meta1", "meta2"],
        model='jeder zottel ist ein model')

    res = w.Response()
    w.PREPARE_DICT = {'id1': w.Condition(None, True)}

    mock_background_task = mocker.patch.object(fastapi.BackgroundTasks, "add_task")
    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = True

    resp_json1 = await w.prepare_json(prepare, mock_background_task, res)

    assert res.status_code == 423
    assert resp_json1.message == "job in progress"


@pytest.mark.asyncio
async def test_predict_status_happy_case(mocker):
    response = w.Response()

    prediction_data = w.PredictionData(
        visit_id=["a.de"],
        category=["cat1"]
    )

    mock_parquet_to_json = mocker.patch.object(w, "parquet_to_json")
    mock_parquet_to_json.return_value = prediction_data

    w.PREDICTION_DICT = {
        "job1": w.Condition(
            err=None,
            work_in_progress=False
        )}

    result = await w.get_predict("job1", response)

    assert result.job_id == "job1"
    assert result.message == "job done"
    assert result.data == prediction_data
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_status_job_not_found(mocker):
    mock_parquet_to_json = mocker.patch.object(w, "parquet_to_json")
    mock_parquet_to_json.return_value = "fake_data"

    response = w.Response()

    result = await w.get_predict("job1", response)

    assert result.job_id == "job1"
    assert result.message == "job not found"
    assert result.data is None
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_predict_status_job_in_progress(mocker):
    w.PREDICTION_DICT = {
        "job1": w.Condition(
            err=None,
            work_in_progress=True
        )}

    mock_parquet_to_json = mocker.patch.object(w, "parquet_to_json")
    mock_parquet_to_json.return_value = "fake_data"

    response = w.Response()

    result = await w.get_predict("job1", response)

    assert result.job_id == "job1"
    assert result.message == "job in progress"
    assert result.data is None
    assert response.status_code == 423


@pytest.mark.asyncio
async def test_predict_status_no_parquet_data(mocker):
    w.PREDICTION_DICT = {
        "job1": w.Condition(
            err=None,
            work_in_progress=False
        )}

    mock_parquet_to_json = mocker.patch.object(w, "parquet_to_json")
    mock_parquet_to_json.return_value = None

    response = w.Response()

    result = await w.get_predict("job1", response)

    assert result.job_id == "job1"
    assert result.message == "job is finished, but could not find result data"
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_predict_status_error(mocker):
    w.PREDICTION_DICT = {
        "job1": w.Condition(
            err=Exception("test"),
            work_in_progress=False
        )}

    mock_parquet_to_json = mocker.patch.object(w, "parquet_to_json")
    mock_parquet_to_json.return_value = None

    response = w.Response()

    result = await w.get_predict("job1", response)

    assert result.job_id == "job1"
    assert result.message.startswith("something went wrong:")
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_predict_happy_case(mocker):
    w.PREDICTION_DICT = {
        "job1": w.Condition(
            err=None,
            work_in_progress=False
        )}

    mock_background_task = mocker.patch.object(w.BackgroundTasks, "add_task")
    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = True

    response = w.Response()

    predict = w.Predict(
        model="jeder zottel ist ein model",
        job_id="job1",
    )

    result = await w.predict(predict, mock_background_task, response)

    assert result.job_id == "job1"
    assert result.message == "job started"
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_predict_model_not_found(mocker):
    w.PREDICTION_DICT = {
        "job1": w.Condition(
            err=None,
            work_in_progress=False
        )}

    mock_background_task = mocker.patch.object(w.BackgroundTasks, "add_task")
    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = False

    response = w.Response()

    predict = w.Predict(
        model="jeder zottel ist ein model",
        job_id="job1",
    )

    result = await w.predict(predict, mock_background_task, response)

    assert result.job_id == "job1"
    assert result.message == "model not found"
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_predict_job_in_progress(mocker):
    w.PREDICTION_DICT = {
        "job1": w.Condition(
            err=None,
            work_in_progress=True
        )}

    mock_background_task = mocker.patch.object(w.BackgroundTasks, "add_task")
    mock_exits = mocker.patch.object(w.Path, "exists")
    mock_exits.return_value = True

    response = w.Response()

    predict = w.Predict(
        model="jeder zottel ist ein model",
        job_id="job1",
    )

    result = await w.predict(predict, mock_background_task, response)

    assert result.job_id == "job1"
    assert result.message == "job in progress"
    assert response.status_code == 423


@pytest.mark.asyncio
async def test_get_models(mocker):
    mock_listdir = mocker.patch.object(w, "listdir")
    mock_listdir.return_value = ["model1.mod", "model2.mod"]

    result = await w.get_models()

    assert result["files"] == ["model1.mod", "model2.mod"]


@pytest.mark.asyncio
async def test_upload_model_happy_case(mocker):
    response = w.Response()

    mock_UploadFile = mocker.patch.object(fastapi.UploadFile, "read")
    mock_UploadFile.file.read.return_value = "fake_data"

    m = mock_open()

    mocker.patch.object(w, "open", m)

    result = await w.upload_model(response, mock_UploadFile)
    handle = m()

    handle.write.assert_called_with('fake_data')
    assert response.status_code == 201
    assert result["message"].startswith("successfully uploaded model")


@pytest.mark.asyncio
async def test_upload_model_content_error(mocker):
    response = w.Response()

    mock_UploadFile = mocker.patch.object(fastapi.UploadFile, "read")
    mock_UploadFile.file.read.side_effect = Exception()

    m = mock_open()

    mocker.patch.object(w, "open", m)

    result = await w.upload_model(response, mock_UploadFile)

    assert response.status_code == 500
    assert result["message"].startswith("there was an error uploading the model")


@pytest.mark.asyncio
async def test_upload_model_write_error(mocker):
    response = w.Response()

    mock_UploadFile = mocker.patch.object(fastapi.UploadFile, "read")
    mock_UploadFile.file.read.return_value = "fake_data"

    mock_open = mocker.patch.object(w, "open")
    mock_open.side_effect = Exception()

    result = await w.upload_model(response, mock_UploadFile)

    assert response.status_code == 500
    assert result["message"].startswith("there was an error uploading the model")


@pytest.mark.asyncio
async def test_delete_models_happy_case(mocker):
    mocker.patch.object(w.Path, "unlink")

    response = w.Response()

    result = await w.delete_models("file1", response)

    assert result["message"] == "deleted successfully"
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_delete_models_error(mocker):
    mocker.patch.object(w.Path, "unlink").side_effect = Exception()

    response = w.Response()

    result = await w.delete_models("file1", response)

    assert result["message"] == "there was an error deleting this model"
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_get_parquet_input_file(mocker):
    mock_listdir = mocker.patch.object(w, "listdir")
    mock_listdir.return_value = ["file1.parquet", "file2.parquet"]

    result = await w.get_input_files()

    assert result["files"] == ["file1.parquet", "file2.parquet"]


@pytest.mark.asyncio
async def test_delete_parquet_input_file_happy_case(mocker):
    mocker.patch.object(w.Path, "unlink")

    response = w.Response()

    result = await w.delete_input_files("file1", response)

    assert result["message"] == "deleted successfully"
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_delete_parquet_input_file_error(mocker):
    mocker.patch.object(w.Path, "unlink").side_effect = Exception()

    response = w.Response()

    result = await w.delete_input_files("file1", response)

    assert result["message"] == "there was an error deleting this parquet file"
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_get_parquet_output_file(mocker):
    mock_listdir = mocker.patch.object(w, "listdir")
    mock_listdir.return_value = ["file1.parquet", "file2.parquet"]

    result = await w.get_output_files()

    assert result["files"] == ["file1.parquet", "file2.parquet"]


@pytest.mark.asyncio
async def test_delete_parquet_output_file_happy_case(mocker):
    mocker.patch.object(w.Path, "unlink")

    response = w.Response()

    result = await w.delete_output_files("file1", response)

    assert result["message"] == "deleted successfully"
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_delete_parquet_output_file_error(mocker):
    mocker.patch.object(w.Path, "unlink").side_effect = Exception()

    response = w.Response()

    result = await w.delete_output_files("file1", response)

    assert result["message"] == "there was an error deleting this parquet file"
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_get_hdf5_file(mocker):
    mock_listdir = mocker.patch.object(w, "listdir")
    mock_listdir.return_value = ["file1.hdf5", "file2.hdf5"]

    result = await w.get_hdf5_files()

    assert result["files"] == ["file1.hdf5", "file2.hdf5"]


@pytest.mark.asyncio
async def test_delete_hdf5_file_happy_case(mocker):
    mocker.patch.object(w.Path, "unlink")

    response = w.Response()

    result = await w.delete_hdf5_files("file1", response)

    assert result["message"] == "deleted successfully"
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_delete_hdf5_file_error(mocker):
    mocker.patch.object(w.Path, "unlink").side_effect = Exception()

    response = w.Response()

    result = await w.delete_hdf5_files("file1", response)

    assert result["message"] == "there was an error deleting this hd5 file"
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_get_job_happy_case():
    w.PREDICTION_DICT = {
        "job1": w.Condition(
            err=None,
            work_in_progress=False
        ),
        "job3": w.Condition(
            err=None,
            work_in_progress=False
        ),

    }

    w.PREPARE_DICT = {
        "job2": w.Condition(
            err=None,
            work_in_progress=False
        ),
        "job4": w.Condition(
            err=None,
            work_in_progress=True
        )
    }

    result = await w.get_jobs()

    assert result["jobs"]["prepare"] == w.PREPARE_DICT
    assert result["jobs"]["predict"] == w.PREDICTION_DICT


@pytest.mark.asyncio
async def test_delete_job_happy_case(mocker):
    w.PREDICTION_DICT = {
        "job1": w.Condition(
            err=None,
            work_in_progress=False
        )}

    w.PREPARE_DICT = {
        "job1": w.Condition(
            err=None,
            work_in_progress=False
        )}

    mock_listdir = mocker.patch.object(w, "listdir")
    mock_listdir.return_value = ["job1.parquet", "job1.hdf5"]

    response = w.Response()
    mocker.patch.object(w.Path, "unlink")

    assert "job1" in w.PREPARE_DICT
    assert "job1" in w.PREDICTION_DICT

    result = await w.delete_job("job1", response)

    assert "job1" not in w.PREPARE_DICT
    assert "job1" not in w.PREDICTION_DICT
    assert result["deleted"] == ['input_file: job1.parquet', 'output_file: job1.parquet', 'hdf5_file: job1.hdf5']
    assert result["message"].startswith("removed all files for job")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_delete_job_predict_running(mocker):
    w.PREDICTION_DICT = {
        "job1": w.Condition(
            err=None,
            work_in_progress=True
        )}
    mock_listdir = mocker.patch.object(w, "listdir")
    mock_listdir.return_value = ["job1.parquet", "job1.hdf5"]

    response = w.Response()
    mocker.patch.object(w.Path, "unlink")

    result = await w.delete_job("job1", response)

    assert result["deleted"] == []
    assert result["message"].startswith("job in progress for")
    assert (response.status_code == 423)


@pytest.mark.asyncio
async def test_delete_job_prepare_running(mocker):
    w.PREPARE_DICT = {
        "job1": w.Condition(
            err=None,
            work_in_progress=True
        )}
    mock_listdir = mocker.patch.object(w, "listdir")
    mock_listdir.return_value = ["job1.parquet", "job1.hdf5"]

    response = w.Response()
    mocker.patch.object(w.Path, "unlink")

    result = await w.delete_job("job1", response)

    assert result["deleted"] == []
    assert result["message"].startswith("job in progress for")
    assert response.status_code == 423


@pytest.mark.asyncio
async def test_delete_job_unlink_exception(mocker):
    mocker.patch.object(w.Path, "unlink").side_effect = Exception()

    mock_listdir = mocker.patch.object(w, "listdir")
    mock_listdir.return_value = ["job1.parquet", "job1.hdf5"]

    response = w.Response()

    result = await w.delete_job("job1", response)

    assert result["deleted"] == []
    assert result["error"].startswith("could not delete file")
    assert response.status_code == 500


def test_do_preparation_happy_case(mocker):
    mocker.patch.object(w, "dit_data_preparation")

    w.do_preparation("job1", "test.model")

    assert "job1" in w.PREPARE_DICT

    assert w.PREPARE_DICT["job1"].err is None
    assert w.PREPARE_DICT["job1"].work_in_progress is False


def test_do_preparation_error(mocker):
    mocker.patch.object(w, "dit_data_preparation").side_effect = Exception()

    w.do_preparation("job1", "test.model")

    assert "job1" in w.PREPARE_DICT

    assert w.PREPARE_DICT["job1"].err is not None
    assert w.PREPARE_DICT["job1"].work_in_progress is False


def test_dit_data_preparation(mocker):
    mock_load = mocker.patch.object(w.torch, "load")
    mock_parquet = mocker.patch.object(w.pq, "ParquetFile")
    mock_preprocess_x = mocker.patch.object(w, "preprocess_x")

    model_path = "model_path"
    parquet_path = "parquet_path"
    hd5_path = "parquet_path"

    w.dit_data_preparation(model_path, parquet_path, hd5_path)

    mock_load.assert_called_once()
    mock_parquet.assert_called_once()
    mock_preprocess_x.assert_called_once()


def test_do_prediction_happy_case(mocker):
    mock_dit_category_prediction = mocker.patch.object(w, "dit_category_prediction")

    predict = w.Predict(
        model="jeder zottel ist ein model",
        job_id="job1",
    )

    w.do_prediction(predict)

    assert "job1" in w.PREDICTION_DICT

    assert w.PREDICTION_DICT["job1"].err is None
    assert w.PREDICTION_DICT["job1"].work_in_progress is False
    mock_dit_category_prediction.assert_called_once()


def test_do_prediction_error(mocker):
    mocker.patch.object(w, "dit_category_prediction").side_effect = Exception()

    predict = w.Predict(
        model="jeder zottel ist ein model",
        job_id="job1",
    )

    w.do_prediction(predict)

    assert "job1" in w.PREDICTION_DICT

    assert w.PREDICTION_DICT["job1"].err is not None
    assert w.PREDICTION_DICT["job1"].work_in_progress is False


def test_dit_category_prediction(mocker):
    class fakeLabelEncoder:
        classes_ = [1, 2]

    mock_load_model = mocker.patch.object(w, "load_model").return_value = 1, 2, 3, fakeLabelEncoder()
    mock_load = mocker.patch.object(w.PreprocessedInputs, "load")
    mock_make_predictions = mocker.patch.object(w, "make_predictions")

    w.dit_category_prediction("a", "b", "c")

    mock_load.assert_called_once()
    mock_make_predictions.assert_called_once()


def test_write_preparation_parquet_from_body():
    prepare = w.PrepareJson(
        job_id="job1",
        domain_name=["test1.de", "test2.de"],
        body_text=["testdata for test1", "testdata for test2"],
        meta_text=["meta1", "meta2"],
        model='jeder zottel ist ein model',
        visit_id=[],
        external_hosts=[]
    )

    w.write_preparation_parquet_from_body(prepare)

    file_path = f'{w.PARQUET_INPUT_PATH}/{prepare.job_id}.parquet'

    df = pandas.read_parquet(file_path)

    assert df["visit_id"].array == prepare.visit_id
    assert df["domain_name"].array == prepare.domain_name
    assert df["body_text"].array == prepare.body_text
    assert df["meta_text"].array == prepare.meta_text
    assert df["external_hosts"].array == prepare.external_hosts

    pathlib.Path(file_path).unlink()


def test_write_preparation_parquet_diff_body_text():
    prepare = w.PrepareJson(
        job_id="job1",
        domain_name=["test1.de", "test2.de"],
        body_text=["testdata for test2"],
        meta_text=["meta1", "meta2"],
        model='jeder zottel ist ein model',
        visit_id=[],
        external_hosts=[]
    )

    with pytest.raises(ValueError, match="domain_name and body_text diff length"):
        w.write_preparation_parquet_from_body(prepare)


def test_write_preparation_diff_meta_text_meta_text():
    prepare = w.PrepareJson(
        job_id="job1",
        domain_name=["test1.de", "test2.de"],
        body_text=["aaaa", "testdata for test2"],
        meta_text=["meta2"],
        model='jeder zottel ist ein model',
        visit_id=[],
        external_hosts=[]
    )

    with pytest.raises(ValueError, match="domain_name and meta_text diff length"):
        w.write_preparation_parquet_from_body(prepare)


def test_write_preparation_parquet_diff_visit_id():
    prepare = w.PrepareJson(
        job_id="job1",
        domain_name=["test1.de", "test2.de"],
        body_text=["testdata for test1", "testdata for test2"],
        meta_text=["meta1", "meta2"],
        model='jeder zottel ist ein model',
        visit_id=["visit_id"],
        external_hosts=[]
    )

    with pytest.raises(ValueError, match="domain_name and visit_id diff length"):
        w.write_preparation_parquet_from_body(prepare)


def test_write_preparation_external_hosts():
    prepare = w.PrepareJson(
        job_id="job1",
        domain_name=["test1.de", "test2.de"],
        body_text=["testdata for test1", "testdata for test2"],
        meta_text=["meta1", "meta2"],
        model='jeder zottel ist ein model',
        visit_id=[],
        external_hosts=["aaa"]
    )

    with pytest.raises(ValueError, match="domain_name and external_hosts diff length"):
        w.write_preparation_parquet_from_body(prepare)
