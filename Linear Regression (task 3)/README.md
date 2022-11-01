<b>Задание 3</b>

В задачах используйте модель линейной регрессии из библиотеки sklearn:

from sklearn.linear_model import LinearRegression

Обучение модели выполняйте с настройками по умолчанию (с аргументами по умолчанию при создании объекта класса LinearRegression):  LinearRegression().fit(X, y)

Перед вами результаты наблюдений длительности нахождения человека в очереди в зависимости от количества людей в этой очереди.

![изображение](https://user-images.githubusercontent.com/39648424/199015401-980b5dd7-4a8e-42d1-94d8-f002ce8f1805.png)

Обучите модель линейной регрессии для прогнозирования и найдите указанные параметры.

<b>1. Выборочное среднее ![изображение](https://user-images.githubusercontent.com/39648424/199015861-bc5c8981-f60a-47ec-9c0a-f35442fa327f.png)</b>

<b>2. Выборочное среднее ![изображение](https://user-images.githubusercontent.com/39648424/199016719-9ea88d4b-adbb-4f06-94a6-f9cdf60d7a2d.png)</b>

<b>3. Коэффициент ![изображение](https://user-images.githubusercontent.com/39648424/199017250-da532d93-f27d-4d51-b612-9fd3c90c70db.png)</b>

<b>4. Коэффициент ![изображение](https://user-images.githubusercontent.com/39648424/199017292-e10be347-f75f-4226-9776-4bf11e903357.png)</b>

<b>5. Оцените точность модели, вычислив ![изображение](https://user-images.githubusercontent.com/39648424/199017432-af3610fe-eba1-49e0-9d56-00e5217e7493.png) статистику</b>

В прилагаемом файле (candy-data.csv) представлены данные, собранные путем голосования за самые лучшие (или, по крайней мере, самые популярные) конфеты Хэллоуина. Обучите модель линейной многомерной регрессии. В качестве предикторов выступают поля: chocolate, fruity, caramel, peanutyalmondy, nougat, crispedricewafer, hard, bar, pluribus, sugarpercent, pricepercent, отклик — winpercent.

В качестве тренировочного набора данных используйте данные из файла, за иключением следующих конфет: Dum Dums, Nestle Smarties. Обучите модель.

<b>6. Найдите предсказанное значение winpercent для конфеты Dum Dums</b>

<b>7. Найдите предсказанное значение winpercent для конфеты Nestle Smarties</b> 

<b>8. Найдите предсказанное значение winpercent для конфеты с параметрами: ![изображение](https://user-images.githubusercontent.com/39648424/199017754-d78e12b4-29e3-49e7-9982-3dc0e94ba724.png)</b>