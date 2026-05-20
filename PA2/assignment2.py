import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics


# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass

def _as_float(x):
    return None if x is None else float(x)

def _as_int(x):
    return 0 if x is None else int(x)

def _null_count(col_name):
    return F.sum(F.when(F.col(col_name).isNull(), 1).otherwise(0))

# -----------------------------------------------------------------------------


def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    ratings = (
        review_data
        .groupBy(asin_column)
        .agg(
            F.avg(F.col(overall_column)).alias(mean_rating_column),
            F.count(F.col(overall_column)).alias(count_rating_column)
        )
    )

    transformed = product_data.join(ratings, on=asin_column, how='left')
    
    stats = transformed.agg(
        F.count("*").alias("count_total"),
        F.avg(mean_rating_column).alias("mean_meanRating"),
        F.variance(mean_rating_column).alias("variance_meanRating"),
        _null_count(mean_rating_column).alias("numNulls_meanRating"),
        F.avg(count_rating_column).alias("mean_countRating"),
        F.variance(count_rating_column).alias("variance_countRating"),
        _null_count(count_rating_column).alias("numNulls_countRating")
    ).first()
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:
    res = {
        'count_total': _as_int(stats["count_total"]),
        'mean_meanRating': _as_float(stats["mean_meanRating"]),
        'variance_meanRating': _as_float(stats["variance_meanRating"]),
        'numNulls_meanRating': _as_int(stats["numNulls_meanRating"]),
        'mean_countRating': _as_float(stats["mean_countRating"]),
        'variance_countRating': _as_float(stats["variance_countRating"]),
        'numNulls_countRating': _as_int(stats["numNulls_countRating"])
    }
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    raw_category = F.col(categories_column).getItem(0).getItem(0)

    transformed = (
        product_data
        .withColumn(
            category_column,
            F.when(
                (F.col(categories_column).isNotNull()) &
                (F.size(F.col(categories_column)) > 0) &
                (F.col(categories_column).getItem(0).isNotNull()) &
                (F.size(F.col(categories_column).getItem(0)) > 0) &
                (raw_category.isNotNull()) &
                (F.trim(raw_category) != ""),
                raw_category
            ).otherwise(F.lit(None).cast(T.StringType()))
        )
        .withColumn(
            bestSalesCategory_column,
            F.when(
                (F.col(salesRank_column).isNotNull()) &
                (F.size(F.col(salesRank_column)) > 0),
                F.map_keys(F.col(salesRank_column)).getItem(0)
            ).otherwise(F.lit(None).cast(T.StringType()))
        )
        .withColumn(
            bestSalesRank_column,
            F.when(
                (F.col(salesRank_column).isNotNull()) &
                (F.size(F.col(salesRank_column)) > 0),
                F.map_values(F.col(salesRank_column)).getItem(0)
            ).otherwise(F.lit(None).cast(T.IntegerType()))
        )
    )

    stats = transformed.agg(
        F.count("*").alias("count_total"),
        F.avg(bestSalesRank_column).alias("mean_bestSalesRank"),
        F.variance(bestSalesRank_column).alias("variance_bestSalesRank"),
        _null_count(category_column).alias("numNulls_category"),
        F.countDistinct(category_column).alias("countDistinct_category"),
        _null_count(bestSalesCategory_column).alias("numNulls_bestSalesCategory"),
        F.countDistinct(bestSalesCategory_column).alias("countDistinct_bestSalesCategory")
    ).first()
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:
        res = {
        'count_total': _as_int(stats["count_total"]),
        'mean_bestSalesRank': _as_float(stats["mean_bestSalesRank"]),
        'variance_bestSalesRank': _as_float(stats["variance_bestSalesRank"]),
        'numNulls_category': _as_int(stats["numNulls_category"]),
        'countDistinct_category': _as_int(stats["countDistinct_category"]),
        'numNulls_bestSalesCategory': _as_int(stats["numNulls_bestSalesCategory"]),
        'countDistinct_bestSalesCategory': _as_int(stats["countDistinct_bestSalesCategory"])
    }
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    rid_column = "__rid"
    also_viewed_column = "__also_viewed"
    viewed_asin_column = "__viewed_asin"
    viewed_price_column = "__viewed_price"

    base = (
        product_data
        .withColumn(rid_column, F.monotonically_increasing_id())
        .withColumn(also_viewed_column, F.col(related_column).getItem(attribute))
        .withColumn(
            countAlsoViewed_column,
            F.when(
                (F.col(also_viewed_column).isNotNull()) &
                (F.size(F.col(also_viewed_column)) > 0),
                F.size(F.col(also_viewed_column))
            ).otherwise(F.lit(None).cast(T.IntegerType()))
        )
    )

    exploded = (
        base
        .where(
            (F.col(also_viewed_column).isNotNull()) &
            (F.size(F.col(also_viewed_column)) > 0)
        )
        .select(
            rid_column,
            F.explode(F.col(also_viewed_column)).alias(viewed_asin_column)
        )
    )

    prices = (
        product_data
        .select(
            F.col(asin_column).alias(viewed_asin_column),
            F.col(price_column).alias(viewed_price_column)
        )
        .where(F.col(viewed_price_column).isNotNull())
    )

    mean_prices = (
        exploded
        .join(F.broadcast(prices), on=viewed_asin_column, how='left')
        .groupBy(rid_column)
        .agg(F.avg(viewed_price_column).alias(meanPriceAlsoViewed_column))
    )

    transformed = base.join(mean_prices, on=rid_column, how='left')
    
    stats = transformed.agg(
        F.count("*").alias("count_total"),
        F.avg(meanPriceAlsoViewed_column).alias("mean_meanPriceAlsoViewed"),
        F.variance(meanPriceAlsoViewed_column).alias("variance_meanPriceAlsoViewed"),
        _null_count(meanPriceAlsoViewed_column).alias("numNulls_meanPriceAlsoViewed"),
        F.avg(countAlsoViewed_column).alias("mean_countAlsoViewed"),
        F.variance(countAlsoViewed_column).alias("variance_countAlsoViewed"),
        _null_count(countAlsoViewed_column).alias("numNulls_countAlsoViewed")
    ).first()
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:
    res = {
        'count_total': _as_int(stats["count_total"]),
        'mean_meanPriceAlsoViewed': _as_float(stats["mean_meanPriceAlsoViewed"]),
        'variance_meanPriceAlsoViewed': _as_float(stats["variance_meanPriceAlsoViewed"]),
        'numNulls_meanPriceAlsoViewed': _as_int(stats["numNulls_meanPriceAlsoViewed"]),
        'mean_countAlsoViewed': _as_float(stats["mean_countAlsoViewed"]),
        'variance_countAlsoViewed': _as_float(stats["variance_countAlsoViewed"]),
        'numNulls_countAlsoViewed': _as_int(stats["numNulls_countAlsoViewed"])
    }
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    price_float_column = "__price_float"

    price_df = product_data.withColumn(
        price_float_column,
        F.col(price_column).cast(T.FloatType())
    )
    
    mean_price = price_df.agg(F.avg(price_float_column).alias("mean_price")).first()["mean_price"]

    median_list = (
        price_df
        .where(F.col(price_float_column).isNotNull())
        .approxQuantile(price_float_column, [0.5], 0.0)
    )
    median_price = median_list[0] if len(median_list) > 0 else None
    
    transformed = (
        price_df
        .withColumn(
            meanImputedPrice_column,
            F.when(F.col(price_float_column).isNull(), F.lit(mean_price))
             .otherwise(F.col(price_float_column))
             .cast(T.FloatType())
        )
        .withColumn(
            medianImputedPrice_column,
            F.when(F.col(price_float_column).isNull(), F.lit(median_price))
             .otherwise(F.col(price_float_column))
             .cast(T.FloatType())
        )
        .withColumn(
            unknownImputedTitle_column,
            F.when(
                F.col(title_column).isNull() | (F.trim(F.col(title_column)) == ""),
                F.lit("unknown")
            ).otherwise(F.col(title_column))
        )
    )
    
    stats = transformed.agg(
        F.count("*").alias("count_total"),
        F.avg(meanImputedPrice_column).alias("mean_meanImputedPrice"),
        F.variance(meanImputedPrice_column).alias("variance_meanImputedPrice"),
        _null_count(meanImputedPrice_column).alias("numNulls_meanImputedPrice"),
        F.avg(medianImputedPrice_column).alias("mean_medianImputedPrice"),
        F.variance(medianImputedPrice_column).alias("variance_medianImputedPrice"),
        _null_count(medianImputedPrice_column).alias("numNulls_medianImputedPrice"),
        F.sum(
            F.when(F.col(unknownImputedTitle_column) == "unknown", 1).otherwise(0)
        ).alias("numUnknowns_unknownImputedTitle")
    ).first()
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:
    res = {
        'count_total': _as_int(stats["count_total"]),
        'mean_meanImputedPrice': _as_float(stats["mean_meanImputedPrice"]),
        'variance_meanImputedPrice': _as_float(stats["variance_meanImputedPrice"]),
        'numNulls_meanImputedPrice': _as_int(stats["numNulls_meanImputedPrice"]),
        'mean_medianImputedPrice': _as_float(stats["mean_medianImputedPrice"]),
        'variance_medianImputedPrice': _as_float(stats["variance_medianImputedPrice"]),
        'numNulls_medianImputedPrice': _as_int(stats["numNulls_medianImputedPrice"]),
        'numUnknowns_unknownImputedTitle': _as_int(stats["numUnknowns_unknownImputedTitle"])
    }
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    product_processed_data_output = product_processed_data.withColumn(
        titleArray_column,
        F.split(F.lower(F.col(title_column)), " ")
    )

    word2vec = M.feature.Word2Vec(
        minCount=100,
        vectorSize=16,
        seed=SEED,
        numPartitions=4,
        inputCol=titleArray_column,
        outputCol=titleVector_column
    )
    
    model = word2vec.fit(product_processed_data_output)
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:
    res['count_total'] = product_processed_data_output.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------
    indexer = M.feature.StringIndexer(
        inputCol=category_column,
        outputCol=categoryIndex_column
    )

    indexed = indexer.fit(product_processed_data).transform(product_processed_data)

    encoder = M.feature.OneHotEncoder(
            inputCols=[categoryIndex_column],
            outputCols=[categoryOneHot_column],
            dropLast=False
        )
    encoded = encoder.fit(indexed).transform(indexed)

    pca = M.feature.PCA(
        k=15,
        inputCol=categoryOneHot_column,
        outputCol=categoryPCA_column
    )

    transformed = pca.fit(encoded).transform(encoded)

    mean_onehot = (
        transformed
        .select(M.stat.Summarizer.mean(F.col(categoryOneHot_column)).alias("mean"))
        .first()["mean"]
    )

    mean_pca = (
        transformed
        .select(M.stat.Summarizer.mean(F.col(categoryPCA_column)).alias("mean"))
        .first()["mean"]
    )
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:
    res = {
        'count_total': int(transformed.count()),
        'meanVector_categoryOneHot': [float(x) for x in mean_onehot.toArray()],
        'meanVector_categoryPCA': [float(x) for x in mean_pca.toArray()]
    }
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    dt = DecisionTreeRegressor(
        featuresCol="features",
        labelCol="overall",
        maxDepth=5
    )

    model = dt.fit(train_data)
    predictions = model.transform(test_data)

    evaluator = RegressionEvaluator(
        labelCol="overall",
        predictionCol="prediction",
        metricName="rmse"
    )

    test_rmse = evaluator.evaluate(predictions)
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:
    res = {
        'test_rmse': float(test_rmse)
    }
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    train_split, valid_split = train_data.randomSplit([0.75, 0.25], seed=SEED)

    evaluator = RegressionEvaluator(
        labelCol="overall",
        predictionCol="prediction",
        metricName="rmse"
    )

    depths = [5, 7, 9, 12]
    models = {}
    valid_rmses = {}

    for depth in depths:
        dt = DecisionTreeRegressor(
            featuresCol="features",
            labelCol="overall",
            maxDepth=depth
        )

        model = dt.fit(train_split)
        models[depth] = model

        valid_predictions = model.transform(valid_split)
        valid_rmses[depth] = float(evaluator.evaluate(valid_predictions))

    best_depth = min(valid_rmses, key=valid_rmses.get)
    best_model = models[best_depth]

    test_predictions = best_model.transform(test_data)
    test_rmse = float(evaluator.evaluate(test_predictions))
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:
    res = {
        'test_rmse': test_rmse,
        'valid_rmse_depth_5': valid_rmses[5],
        'valid_rmse_depth_7': valid_rmses[7],
        'valid_rmse_depth_9': valid_rmses[9],
        'valid_rmse_depth_12': valid_rmses[12],
    }
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------

