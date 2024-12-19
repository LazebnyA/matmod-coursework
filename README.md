# Курсова робота з Методів Оптимізації

## Тема

### Моделювання росту чисельності ізольованих популяцій. Модель Хатчинсона

## Вступ

### Мета

### Змістовна постановка задачі

Розробити математичну модель, що дозволяє описати процес зростання чисельності ізольованої популяції.

Модель має дозволяти:

- Обчислювати чисельність популяції будь-якої миті часу з заданого часового проміжку.

- Будувати графіки чисельності популяції для різних значень параметрів.

Вхідні дані:

- $t_0$ - початок часового проміжку

- $T$ - кінець часового проміжку

- $r > 0$ $-$ коефіцієнт лінійного зростання,

- $K > 0$ $-$ ємність середовища,

- $\tau > 0$ $-$ запізнення.

### Концептуальна постановка задачі

Зростання чисельнності ізольованих популяцій можна змоделювати відповідно до *моделі Хатчинсона*: модифікації логістичної моделі, я рамках якої врахований ефект *запізнення*.

Приймемо такі гіпотези:

- Об'єктом моделювання є чисельність популяції з плином часу $N(t)$. Від початкового моменту часу $t_0$ до кінцевого - $T$.

- Середовищем в якому існує популяція має ємність $K$, що визначається доступною кількістю ресурсів.

- Під час зростання чисельності популяції має місце запізнення - поточний стан популяції впливає на її динаміку через певний часовий проміжок 
$\tau$. Це враховує такі природні фактори, як час статевого дозрівання або затримку між моментом зачаття і народженням. 

- Нехтуємо непередбачуваними зовнішніми або внутрішніми факторами затримки/збільшення чисельності популяції (наприклад, війни, стихійні лиха, епідемії). Єдиним фактором, що враховується - є запізнення.

Відповідно до вищеописаних гіпотез **концептуальну постановку задачі** можна представити у такому вигляді:

*Визначити функцію зростання ізольованої популяції $N(t)$ на обмеженому часовому проміжку від $t_0$ до $T$, враховуючи що відомі такі показники як: коефіцієнт лінійного зростання $r$, ємність середовища $K$ та коефіцієнт запізнення $\tau$.*

### Математична постановка задачі

Знайти залежність чисельності ізольованої популяції від часу - $N(t)$ - з вирішення диференційного рівняння Хатчинсона:

$$
\frac{dN}{dt} = r\left(1 - \frac{N(t - \tau)}{K}\right) N(t), \quad t \geq 0. \qquad (1)
$$

Початкова умова:

$$
N(t) = \phi(t), \quad t \in \left[ -\tau, 0 \right], \qquad (2)
$$

де $\phi(t) \geq 0$ - неперервна функція.

В даній задачі, за таку функцію було обрано:
$$
\phi(t) = \cos{\left(0.01 t\right)} \cdot N_0
$$

де $N_0$ - початкова чисельність населення


## Основна частина

### Вибір та обґрунтування вибору методу розв'язання задачі

Оскільки аналітично розв'язання заданого в математичній постановці задачі диференційного рівняння знайти складно, досить часто використовують чисельні методи, наприклад, метод Ейлера або метод Рунге-Кутти.

В даній роботі, буде використаний метод Рунге-Кутти 4 порядку. 

Метод Рунге-Кутти 4-го порядку використовується для чисельного інтегрування звичайних диференціальних рівнянь (ODE). Для задач з затримкою, як у нашому випадку, де необхідно враховувати значення 
$N(t - \tau)$, процедура інтегрування складається з таких кроків:

1. Ініціалізація: Для часу $t \le 0$ встановлюється початкова умова:
    $$
    \phi(t) = \cos{\left(0.01 t\right)} \cdot N_0
    $$

2. Визначення коефіцієнтів Рунге-Кутти:
    - Для кожного кроку часу $t_i$ обчислюються чотири коефіцієнти $k_1, k_2, k_3, k_4$, які використовуються для апроксимації розв'язку на наступному кроці.

3. Формули Коефіцієнтів: 
    - $k_1 = f\left(t_i, N_i\right)$
    - $k_2 = f\left(t_i + \frac{dt}{2}, N_i + \frac{dt}{2} \cdot k_1\right)$
    - $k_3 = f\left(t_i + \frac{dt}{2}, N_i + \frac{dt}{2} \cdot k_2\right)$
    - $k_4 = f\left(t_i + dt, N_i + dt \cdot k_3\right)$

    - Формула для оновлення значення $N(t)$ виглядає так:
    $$
    N_{i+1} = N_i + \frac{dt}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)
    $$

4. Робота з затримкою:
    - Для кожного часу $t_i$ використовуються значення популяції $N(t - \tau)$, яке визначається через значення на попередніх кроках (або через початкову умову коли $t - \tau > 0$).


### Програмна реалізація моделі Хатчинсона, порівняння результатів для різних наборів вхідних параметрів та аналіз моделі на адекватність.

> *Знаходиться в **Jupyter Notebook** файлі під назвою coursework.ipynb*

1. **Обрати різноманітні вхідні параметри моделі та проаналізувати особливості отриманих результатів.**

2. **Оцінити адекватність результатів моделювання.**

## Висновки

Зробити висновки щодо виконаної роботи.

## Список використаних джерел

1. Маценко В . Г . Математичне моделювання: навчальний посібник c. 329

