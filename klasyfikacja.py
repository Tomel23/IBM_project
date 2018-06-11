from __future__ import print_function
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString
from pyspark.sql import SparkSession

wml_credentials = {
    "url": "https://ibm-watson-ml.mybluemix.net",
    "username": "b62a0f07-c48e-4671-b47a-61c1b7a09381",
    "password": "26606e9c-2e7f-4142-868c-114ee4ebd718",
    "instance_id": "941ee500-925a-41be-91eb-4d38b008bc32"
}


def connect(wml_credentials):
    from watson_machine_learning_client import WatsonMachineLearningAPIClient
    return WatsonMachineLearningAPIClient(wml_credentials)


def get_model_uid(client):
    model_resources = client.repository.get_model_details()[u'resources']
    model_values = [
        (m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], m[u'entity'][u'model_type'])
        for
        m in model_resources]

    for m in model_values:
        if m[1] == 'Project predict model':
            return m[0]


def get_deployment_url(client):
    details = client.deployments.get_details()
    resources = details[u'resources']
    deployments_values = [
        (m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'entity'][u'type'], m[u'entity'][u'status'],
         m[u'metadata'][u'created_at'], m[u'entity'][u'model_type']) for m in resources]

    for m in deployments_values:
        if m[1] == 'Product predict model deployment':
            return client.deployments.get_scoring_url(client.deployments.get_details(m[0]))


def get_deployment_uid(client):
    details = client.deployments.get_details()
    resources = details[u'resources']
    deployments_values = [
        (m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'entity'][u'type'], m[u'entity'][u'status'],
         m[u'metadata'][u'created_at'], m[u'entity'][u'model_type']) for m in resources]

    for m in deployments_values:
        if m[1] == 'Product predict model deployment':
            return m[0]


def send_request(values, deployment_url):
    import urllib3, requests, json
    headers = urllib3.util.make_headers(basic_auth='{username}:{password}'.format(username=wml_credentials['username'],
                                                                                  password=wml_credentials['password']))
    url = '{}/v3/identity/token'.format(wml_credentials['url'])
    response = requests.get(url, headers=headers)
    mltoken = json.loads(response.text).get('token')

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    # NOTE: manually define and pass the array(t) of values to be scored in the next line
    payload_scoring = {"fields": ["SCALE", "PRICE", "SUBCATEGORY"],
                       "values": [values]}

    response_scoring = requests.post(
        deployment_url,
        json=payload_scoring, headers=header)

    response = json.loads(response_scoring.text)
    # tmp=json.dumps(response_scoring,indent=2)
    print(response)
    lista = sorted(response['values'][0][8], reverse=True)
    id = response['values'][0][10].split(' ')
    probability = lista[0]
    return probability, id[1]


def create_and_deploy_model():
    # Starting spark session
    spark = SparkSession \
        .builder \
        .appName("NaiveBayesExample") \
        .getOrCreate()

    # load data from file
    data = spark.read \
        .format('csv') \
        .option('header', 'true') \
        .option('inferSchema', 'true') \
        .load("Project_List_Data_Set.csv")

    # Split the data into train and test

    splits = data.randomSplit([0.8, 0.2], 1234)
    train = splits[0]
    test = splits[1]

    # Convert text col to integer col and fit data to label col
    stringIndexer_label = StringIndexer(inputCol="ID", outputCol="label").fit(data)
    stringIndexer_scale = StringIndexer(inputCol="SCALE", outputCol="SCALE_X")

    # Converter form int to string for label column
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=stringIndexer_label.labels)

    # nowy wektor z nowymi kolumnami bez id
    vectorAssembler_features = VectorAssembler(inputCols=["SCALE_X", "PRICE", "SUBCATEGORY"], outputCol="features")

    # create the trainer and set its parameters
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

    pipeline = Pipeline(
        stages=[stringIndexer_label, stringIndexer_scale, vectorAssembler_features, nb,
                labelConverter])

    # train the model
    model = pipeline.fit(train)

    '''
    # Checking prediction
    # select example rows to display.
    predictions = model.transform(test)
    predictions.show()
    #calculate accuracy
    accuracy = evaluator.evaluate(predictions)
    print("Test set accuracy = " + str(accuracy))
    '''

    # evaluate model
    # set accuracy as metric
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="accuracy")

    # Set connection to cloud
    client = connect(wml_credentials)

    # Save model in cloud
    published_model_details = client.repository.store_model(model, meta_props={'name': 'Project predict model'},
                                                            training_data=train, pipeline=pipeline)

    model_uid = client.repository.get_model_uid(published_model_details)

    # Spark stop
    spark.stop()

    # Model deploing
    deployment_details = client.deployments.create(model_uid=model_uid, name='Product predict model deployment')


    # scoring_url = client.deployments.get_scoring_url(deployment_details)


def delete_model_and_deployment():
    client = connect(wml_credentials)
    model_uid = get_model_uid(client)
    deployment_uid = get_deployment_uid(client)

    if deployment_uid:
        client.deployments.delete(deployment_uid)

    if model_uid:
        client.repository.delete_model(model_uid)


def check_deployment():
    client = connect(wml_credentials)
    url = get_deployment_url(client)
    if url:
        return url


def check_model():
    client = connect()
    uid = get_model_uid(client)
    if uid:
        return uid


def PredictionView(scale, price, cat):
    # przygotowanie wartosci do predkcji
    values = [scale, price, cat]
    print(values)
    # sprawdzenie czy jest deployment

    deployment = check_deployment()
    if deployment:
        # jesli jest to odpalamy klasyfikator ktory zwraca nam prawdopodobienstwo i id proj
        probability, id = send_request(values, deployment)
        print(id)
        print(probability)
        # odpalam optymalizacje
        # Optymization(1)

    # jeśli nie to tworzymy nowy model(wczesniej nie było zadnego albo ktos usunoł z chmury)
    elif create_and_deploy_model():
        # po utworzeniu odpalamy klasyfikator
        probability, id = send_request(values, deployment)
        # odpalam optymalizacje
        # Optymization(1)
    # jak nie można utworzyc modelu
    else:
        print('error nie mozna utworzyc modelu')


if __name__ == "__main__":
    PredictionView('M', 20, 3, )  # , get_deployment_url(connect(wml_credentials)))

