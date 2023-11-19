"""
통계분석 유틸리티
@Author: 이광호(leekh4232@gmail.com)
"""
import sys
import numpy as np
import seaborn as sb
from pca import pca
from tabulate import tabulate
from matplotlib import pyplot as plt

from pandas import DataFrame, MultiIndex, concat, DatetimeIndex, Series

from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class RegMetric:
    def __init__(self,y,y_pred):
        # 설명력
        self._r2=r2_score(y,y_pred)
        # 평균절대오차
        self._mae=mean_absolute_error(y,y_pred)
        # 평균제곱오차
        self._mse=mean_squared_error(y,y_pred)
        # 평균오차
        self._rmse=np.sqrt(self._mse)

        # 평균 절대 백분오차 비율
        if type(y)==Series:
            self._mape = np.mean(np.abs((y.values - y_pred) / y.values) * 100)
        
        else:
            self._mape = np.mean(np.abs((y - y_pred) / y) * 100)
        
        # 평균비율오차
        if type(y) == Series:   
            self._mpe = np.mean((y.values - y_pred) / y.values * 100)
        else:
            self._mpe = np.mean((y - y_pred) / y * 100)

    @property
    def r2(self):
        return self._r2

    @r2.setter
    def r2(self, value):
        self._r2 = value

    @property
    def mae(self):
        return self._mae

    @mae.setter
    def mae(self, value):
        self._mae = value

    @property
    def mse(self):
        return self._mse

    @mse.setter
    def mse(self, value):
        self._mse = value

    @property
    def rmse(self):
        return self._rmse

    @rmse.setter
    def rmse(self, value):
        self._rmse = value

    @property
    def mape(self):
        return self._mape

    @mape.setter
    def mape(self, value):
        self._mape = value

    @property
    def mpe(self):
        return self._mpe

    @mpe.setter
    def mpe(self, value):
        self._mpe = value

class OlsResult:
    def __init__(self):
        self._x_train = None
        self._y_train = None
        self._train_pred = None
        self._x_test = None
        self._y_test = None
        self._test_pred = None
        self._model = None
        self._fit = None
        self._summary = None
        self._table = None
        self._result = None
        self._goodness = None
        self._varstr = None
        self._coef = None
        self._intercept = None
        self._trainRegMetric = None
        self._testRegMetric = None
    
    @property
    def x_train(self):
        return self._x_train

    @x_train.setter
    def x_train(self, value):
        self._x_train = value

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def train_pred(self):
        return self._train_pred

    @train_pred.setter
    def train_pred(self, value):
        self._train_pred = value

    @property
    def x_test(self):
        return self._x_test

    @x_test.setter
    def x_test(self, value):
        self._x_test = value

    @property
    def y_test(self):
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = value

    @property
    def test_pred(self):
        return self._test_pred

    @test_pred.setter
    def test_pred(self, value):
        self._test_pred = value

    @property
    def model(self):
        """
        분석모델
        """
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def fit(self):
        """
        분석결과 객체
        """
        return self._fit

    @fit.setter
    def fit(self, value):
        self._fit = value

    @property
    def summary(self):
        """
        분석결과 요약 보고
        """
        return self._summary

    @summary.setter
    def summary(self, value):
        self._summary = value

    @property
    def table(self):
        """
        결과표
        """
        return self._table

    @table.setter
    def table(self, value):
        self._table = value

    @property
    def result(self):
        """
        결과표 부가 설명
        """
        return self._result

    @result.setter
    def result(self, value):
        self._result = value

    @property
    def goodness(self):
        """
        모형 적합도 보고
        """
        return self._goodness

    @goodness.setter
    def goodness(self, value):
        self._goodness = value

    @property
    def varstr(self):
        """
        독립변수 보고
        """
        return self._varstr

    @varstr.setter
    def varstr(self, value):
        self._varstr = value
    
    @property
    def coef(self):
        return self._coef

    @coef.setter
    def coef(self, value):
        self._coef = value

    @property
    def intercept(self):
        return self._intercept

    @intercept.setter
    def intercept(self, value):
        self._intercept = value

    @property
    def trainRegMetric(self):
        return self._trainRegMetric

    @trainRegMetric.setter
    def trainRegMetric(self, value):
        self._trainRegMetric = value

    @property
    def testRegMetric(self):
        return self._testRegMetric

    @testRegMetric.setter
    def testRegMetric(self, value):
        self._testRegMetric = value

    def setRegMetric(self, y_train, y_train_pred, y_test=None, y_test_pred=None):
        self.trainRegMetric = RegMetric(y_train, y_train_pred)

        if y_test is not None and y_test_pred is not None:
            self.testRegMetric = RegMetric(y_test, y_test_pred)

def myOls(data, y=None, x=None, expr=None):
    """
    회귀분석을 수행한다.

    Parameters
    -------
    - data : 데이터 프레임
    - y: 종속변수 이름
    - x: 독립변수의 이름들(리스트)
    """

    # 데이터프레임 복사
    df=data.copy()

    # 종속변수~독립변수1+독립변수2+독립변수3+... 형태의 식을 생성
    if not expr:
        #독립변수의 이름이 리스트가 아니라면 리스트로 변환
        if type(x) != list:
            x=[x]
        expr="%s~%s" % (y, "+".join(x))
    
    else:
        x = []
        p = expr.find('~')
        y = expr[:p].strip()
        x_tmp = expr[p+1:]
        x_list = x_tmp.split('+')

        for i in x_list:
            k=i.strip()

            if k:
                x.append(k)

    # 회귀모델 생성
    model = ols(expr, data=data)
    # 분석 수행
    fit = model.fit()

    # 파이썬 분석결과를 변수에 저장한다.
    summary = fit.summary()

    # 회귀 계수 (beta) 추출
    beta_values = fit.params.drop("Intercept")  # Intercept 변수 제외

    # 첫 번째, 세 번째 표의 내용을 딕셔너리로 분해
    my = {}

    for k in range(0, 3, 2):
        items = summary.tables[k].data
        # print(items)

        for item in items:
            # print(item)
            n = len(item)

            for i in range(0, n, 2):
                key = item[i].strip()[:-1]
                value = item[i+1].strip()

                if key and value:
                    my[key] = value

    # 두 번째 표의 내용을 딕셔너리로 분해하여 my에 추가
    my['variables'] = []
    name_list = list(data.columns)
    #print(name_list)

    for i, v in enumerate(summary.tables[1].data):
        if i == 0:
            continue

        # 변수의 이름
        name = v[0].strip()

        vif = 0

        # Intercept는 제외
        if name in name_list:
            # 변수의 이름 목록에서 현재 변수가 몇 번째 항목인지 찾기 
            j = name_list.index(name)
            vif = variance_inflation_factor(data, j)

        my['variables'].append({
            "name": name,
            "coef": v[1].strip(),
            "std err": v[2].strip(),
            "t": v[3].strip(),
            "P-value": v[4].strip(),
            "Beta": beta_values.get(name, 0),
            "VIF": vif,
        })

    # 결과표를 데이터프레임으로 구성
    mylist = []
    yname_list = []
    xname_list = []

    for i in my['variables']:
        if i['name'] == 'Intercept':
            continue

        yname_list.append(y)
        xname_list.append(i['name'])

        item = {
            "B": i['coef'],
            "표준오차": i['std err'],
            "β": i['Beta'],
            "t": "%s*" % i['t'],
            "유의확률": i['P-value'],
            "VIF": i["VIF"]
        }

        mylist.append(item)

    table = DataFrame(mylist,
                   index=MultiIndex.from_arrays([yname_list, xname_list], names=['종속변수', '독립변수']))
    
    # 분석결과
    result = "𝑅(%s), 𝑅^2(%s), 𝐹(%s), 유의확률(%s), Durbin-Watson(%s)" % (my['R-squared'], my['Adj. R-squared'], my['F-statistic'], my['Prob (F-statistic)'], my['Durbin-Watson'])

    # 모형 적합도 보고
    goodness = "%s에 대하여 %s로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 %s(F(%s,%s) = %s, p < 0.05)." % (y, ",".join(x), "유의하다" if float(my['Prob (F-statistic)']) < 0.05 else "유의하지 않다", my['Df Model'], my['Df Residuals'], my['F-statistic'])

    # 독립변수 보고
    varstr = []

    for i, v in enumerate(my['variables']):
        if i == 0:
            continue
        
        s = "%s의 회귀계수는 %s(p%s0.05)로, %s에 대하여 %s."
        k = s % (v['name'], v['coef'], "<" if float(v['P-value']) < 0.05 else '>', y, '유의미한 예측변인인 것으로 나타났다' if float(v['P-value']) < 0.05 else '유의하지 않은 예측변인인 것으로 나타났다')

        varstr.append(k)

    ols_result = OlsResult()
    ols_result.model = model
    ols_result.fit = fit
    ols_result.summary = summary
    ols_result.table = table
    ols_result.result = result
    ols_result.goodness = goodness
    ols_result.varstr = varstr

    return ols_result

def scalling(df, yname=None):
    """
    데이터 프레임을 표준화 한다.

    Parameters
    -------
    - df: 데이터 프레임
    - yname: 종속변수 이름

    Returns
    -------
    - x_train_std_df: 표준화된 독립변수 데이터 프레임
    - y_train_std_df: 표준화된 종속변수 데이터 프레임
    """
    # 평소에는 yname을 제거한 항목을 사용
    # yname이 있지 않다면 df를 복사
    x_train = df.drop([yname], axis=1) if yname else df.copy()
    x_train_std = StandardScaler().fit_transform(x_train)
    x_train_std_df = DataFrame(x_train_std, columns=x_train.columns)
    
    if yname:
        y_train = df.filter([yname])
        y_train_std = StandardScaler().fit_transform(y_train)
        y_train_std_df = DataFrame(y_train_std, columns=y_train.columns)

    if yname:
        result = (x_train_std_df, y_train_std_df)
    else:
        result = x_train_std_df

    return result

class LogitResult:
    def __init__(self):
        self._model = None    
        self._fit = None
        self._summary = None
        self._prs = None
        self._cmdf = None
        self._result_df = None
        self._odds_rate_df = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def fit(self):
        return self._fit

    @fit.setter
    def fit(self, value):
        self._fit = value

    @property
    def summary(self):
        return self._summary

    @summary.setter
    def summary(self, value):
        self._summary = value

    @property
    def prs(self):
        return self._prs

    @prs.setter
    def prs(self, value):
        self._prs = value

    @property
    def cmdf(self):
        return self._cmdf

    @cmdf.setter
    def cmdf(self, value):
        self._cmdf = value

    @property
    def result_df(self):
        return self._result_df

    @result_df.setter
    def result_df(self, value):
        self._result_df = value

    @property
    def odds_rate_df(self):
        return self._odds_rate_df

    @odds_rate_df.setter
    def odds_rate_df(self, value):
        self._odds_rate_df = value
    
def expTimeData(data, yname, sd_model="m", max_diff=1):
    plt.rcParams["font.family"] = 'AppleGothic' if sys.platform == 'darwin' else 'Malgun Gothic'
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.unicode_minus"] = False

    df = data.copy()

    # 데이터 정상성 여부
    stationarity = False

    # 반복 수행 횟수
    count = 0

    # 결측치 존재 여부
    na_count = df[yname].isna().sum()
    print("결측치 수: %d" % na_count)

    plt.figure(figsize=(4, 5))
    sb.boxplot(data=df, y=yname)
    plt.show()
    plt.close()
    
    # 시계열 분해
    model_name = 'multiplicative' if sd_model == 'm' else 'additive'
    sd = seasonal_decompose(df[yname], model=model_name)

    figure = sd.plot()
    figure.set_figwidth(15)
    figure.set_figheight(16)
    fig, ax1, ax2, ax3, ax4 = figure.get_children()
    figure.subplots_adjust(hspace=0.4)

    ax1.set_ylabel("Original")
    ax1.grid(True)
    ax1.title.set_text("Original")
    ax2.grid(True)
    ax2.title.set_text("Trend")
    ax3.grid(True)
    ax3.title.set_text("Seasonal")
    ax4.grid(True)
    ax4.title.set_text("Residual")

    plt.show()

    # ACF, PACF 검정
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.subplots_adjust(hspace=0.4)

    sb.lineplot(data=df, x=df.index, y=yname, ax=ax1)
    ax1.title.set_text("Original")

    plot_acf(df[yname], ax=ax2)
    ax2.title.set_text("ACF Test")
        
    plot_pacf(df[yname], ax=ax3)
    ax3.title.set_text("PACF Test")
        
    plt.show()
    plt.close()

    while not stationarity:
        if count == 0:
            print("=========== 원본 데이터 ===========")
        else:
            print("=========== %d차 차분 데이터 ===========" % count)

        # ADF Test
        ar = adfuller(df[yname])

        ardict = {
            '검정통계량(ADF Statistic)': [ar[0]],
            '유의수준(p-value)': [ar[1]],
            '최적차수(num of lags)': [ar[2]],
            '관측치 개수(num of observations)': [ar[3]]   
        }

        for key, value in ar[4].items():
            ardict['기각값(Critical Values) %s' % key] = value

        stationarity = ar[1] < 0.05
        ardict['데이터 정상성 여부(0=Flase,1=True)'] = stationarity

        ardf = DataFrame(ardict, index=['ADF Test']).T

        print(tabulate(ardf, headers=["ADF Test", ""], tablefmt='psql', numalign="right"))

        # 차분 수행
        df = df.diff().dropna()

        # 반복을 계속할지 여부 판단
        count += 1
        if count == max_diff:
            break

def exp_time_data(data, yname, sd_model="m", max_diff=1):
    expTimeData(data, yname, sd_model, max_diff)
    
def set_datetime_index(df, field=None, inplace=False):
    """
        데이터 프레임의 인덱스를 datetime 형식으로 변환

        Parameters
        -------
        - df: 데이터 프레임
        - inplace: 원본 데이터 프레임에 적용 여부

        Returns
        -------
        - 인덱스가 datetime 형식으로 변환된 데이터 프레임
    """
    
    if inplace:
        if field is not None:
            df.set_index(field, inplace=True)
            
        df.index = DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
        df.sort_index(inplace=True)
    else:
        cdf = df.copy()
        
        if field is not None:
            cdf.set_index(field, inplace=True)
            
        cdf.index = DatetimeIndex(cdf.index.values, freq=cdf.index.inferred_freq)
        cdf.sort_index(inplace=True)
        return cdf
    
def convertPoly(data,degree=2,include_bias=False):
    poly=PolynomialFeatures(degree=degree,include_bias=include_bias)
    fit=poly.fit_transform(data)
    x=DataFrame(fit,columns=poly.get_feature_names_out())
    return x

def getTrend(x,y,degree=2,value_count=100):
    #[a,b,c]==>ax^2+bx+c
    coeff=np.polyfit(x,y,degree)

    if type(x)=='list':
        minx=min(x)
        maxx=max(x)
    else:
        minx=x.min()
        maxx=x.max()

    Vtrend=np.linspace(minx,maxx,value_count)

    Ttrend=coeff[-1]

    for i in range(0,degree):
        Ttrend+=coeff[i]*Vtrend**(degree-i)

    return (Vtrend, Ttrend)

def regplot(x_left, y_left, y_left_pred=None, left_title=None, x_right=None, y_right=None, y_right_pred=None, right_title=None, figsize=(10, 5), save_path=None):
    subcount = 1 if x_right is None else 2
    
    fig, ax = plt.subplots(1, subcount, figsize=figsize)
    
    axmain = ax if subcount == 1 else ax[0]
    
    # 왼쪽 산점도
    sb.scatterplot(x=x_left, y=y_left, label='data', ax=axmain)
    
    # 왼쪽 추세선
    x, y = getTrend(x_left, y_left)
    sb.lineplot(x=x, y=y, color='blue', linestyle="--", ax=axmain)
    
    # 왼쪽 추정치
    if y_left_pred is not None:
        sb.scatterplot(x=x_left, y=y_left_pred, label='predict', ax=axmain)
        # 추정치에 대한 추세선
        x, y = getTrend(x_left, y_left_pred)
        sb.lineplot(x=x, y=y, color='red', linestyle="--", ax=axmain)
    
    if left_title is not None:
        axmain.set_title(left_title)
        
    axmain.legend()
    axmain.grid()
    
    
    if x_right is not None:
        # 오른쪽 산점도
        sb.scatterplot(x=x_right, y=y_right, label='data', ax=ax[1])
        
        # 오른쪽 추세선
        x, y = getTrend(x_right, y_right)
        sb.lineplot(x=x, y=y, color='blue', linestyle="--", ax=ax[1])
    
        # 오른쪽 추정치
        if y_right_pred is not None:
            sb.scatterplot(x=x_right, y=y_right_pred, label='predict', ax=ax[1])
            # 추정치에 대한 추세선
            x, y = getTrend(x_right, y_right_pred)
            sb.lineplot(x=x, y=y, color='red', linestyle="--", ax=ax[1])
        
        if right_title is not None:
            ax[1].set_title(right_title)
            
        ax[1].legend()
        ax[1].grid()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        
    plt.show()
    plt.close()