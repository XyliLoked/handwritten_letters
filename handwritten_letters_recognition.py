"""ОПТИМИЗИРОВАННАЯ СИСТЕМА РАСПОЗНАВАНИЯ РУКОПИСНЫХ АНГЛИЙСКИХ БУКВ С УСКОРЕНИЕМ"""

# ==================== ИМПОРТ БИБЛИОТЕК ====================
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Настройки для отображения
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
print("✅ Библиотеки загружены")

# ==================== ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ====================
print("📥 Загружаем dataset рукописных букв...")

dataset = np.loadtxt('https://storage.yandexcloud.net/academy.ai/A_Z_Handwritten_Data.csv', delimiter=',')

X = dataset[:,1:785]
Y = dataset[:,0]

print(f"📊 Размер dataset: {dataset.shape}")
print(f"📊 Размер X (изображения): {X.shape}")
print(f"📊 Размер Y (метки): {Y.shape}")

# Словарь для преобразования цифр в буквы
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

# ==================== ВИЗУАЛИЗАЦИЯ ДАННЫХ ====================
print("\n👀 Визуализация примеров рукописных букв...")

plt.figure(figsize=(15, 10))
for i in range(40):
    x = X[i]
    x = x.reshape((28, 28))
    plt.subplot(5, 8, i+1)
    plt.imshow(x, cmap='gray')
    plt.title(f'{word_dict.get(Y[i])} ({int(Y[i])})')
    plt.axis('off')
plt.tight_layout()
plt.show()

# ==================== РАЗДЕЛЕНИЕ ДАННЫХ ====================
print("\n🔄 Разделяем данные на train/test...")

(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

print(f"✅ Разделение завершено:")
print(f"   x_train: {x_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   x_test: {x_test.shape}")
print(f"   y_test: {y_test.shape}")

# ==================== ПРЕДОБРАБОТКА ДАННЫХ ====================
print("\n🔄 Подготовка данных...")

# Нормализация пикселей (0-255 -> 0-1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-Hot Encoding для 26 классов букв
y_train_categorical = tf.keras.utils.to_categorical(y_train, 26)
y_test_categorical = tf.keras.utils.to_categorical(y_test, 26)

print("✅ Данные подготовлены:")

# ==================== СОЗДАНИЕ ОПТИМИЗИРОВАННОЙ МОДЕЛИ ====================
def create_optimized_model():
    """Создает оптимизированную модель с улучшенным оптимизатором"""
    model = models.Sequential([
        # Первый скрытый слой
        layers.Dense(1024, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),

        # Второй скрытый слой
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),

        # Третий скрытый слой
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),

        # Четвертый скрытый слой
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        # Выходной слой (26 нейронов для 26 букв)
        layers.Dense(26, activation='softmax')
    ])

    # УЛУЧШЕННАЯ КОМПИЛЯЦИЯ С ОПТИМИЗИРОВАННЫМИ ОПТИМИЗАТОРАМИ
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

print("🧠 Создаем оптимизированную модель с улучшенным Adam...")
model = create_optimized_model()

print("\n📐 Архитектура модели:")
model.summary()

# ==================== ОБУЧЕНИЕ МОДЕЛИ С УСКОРЕНИЕМ ====================
print("\n🎯 Начинаем УСКОРЕННОЕ обучение (5 эпох)...")

# УЛУЧШЕННЫЕ КОЛБЭКИ ДЛЯ УСКОРЕНИЯ
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,  # Уменьшили для быстрого обучения
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,  # Более быстрая реакция
    min_lr=0.0001,
    verbose=1
)

print("⏱️  Обучение на 5 эпохах с оптимизированным Adam...")

history = model.fit(
    x_train,
    y_train_categorical,
    epochs=5,  # ТОЛЬКО 5 ЭПОХ
    batch_size=256,
    validation_data=(x_test, y_test_categorical),
    callbacks=[early_stopping, reduce_lr],  # ⬅️ ВОТ ТУТ УБРАЛИ tensorboard
    verbose=1
)

print("✅ Обучение завершено!")

# ==================== СРАВНЕНИЕ ОПТИМИЗАТОРОВ ====================
print("\n🔬 Тестируем разные оптимизаторы для сравнения...")

optimizers = {
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
    'Nadam': tf.keras.optimizers.Nadam(learning_rate=0.001),
    'Adamax': tf.keras.optimizers.Adamax(learning_rate=0.001)
}

optimizer_results = {}

for opt_name, optimizer in optimizers.items():
    print(f"\n🧪 Тестируем {opt_name}...")

    # Создаем модель с текущим оптимизатором
    test_model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(26, activation='softmax')
    ])

    test_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Быстрое обучение на 3 эпохах
    test_model.fit(
        x_train, y_train_categorical,
        epochs=3,
        batch_size=256,
        verbose=0
    )

    # Оценка
    test_loss, test_accuracy = test_model.evaluate(x_test, y_test_categorical, verbose=0)
    optimizer_results[opt_name] = test_accuracy
    print(f"   {opt_name} точность: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Визуализация сравнения оптимизаторов
plt.figure(figsize=(10, 6))
names = list(optimizer_results.keys())
accuracies = [optimizer_results[name] * 100 for name in names]

bars = plt.bar(names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.title('Сравнение оптимизаторов (3 эпохи)', fontsize=14)
plt.ylabel('Точность (%)')
plt.ylim(80, 95)
plt.grid(True, alpha=0.3)

# Добавляем значения на столбцы
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ==================== АНАЛИЗ ПРОЦЕССА ОБУЧЕНИЯ ====================
print("\n📈 Анализ процесса обучения...")

def plot_training_history(history):
    """Визуализация процесса обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # График потерь
    ax1.plot(history.history['loss'], label='Тренировочные потери', linewidth=2, marker='o')
    ax1.plot(history.history['val_loss'], label='Валидационные потери', linewidth=2, marker='s')
    ax1.set_title('Функция потерь (5 эпох)')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потери')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График точности
    ax2.plot(history.history['accuracy'], label='Тренировочная точность', linewidth=2, marker='o')
    ax2.plot(history.history['val_accuracy'], label='Валидационная точность', linewidth=2, marker='s')
    ax2.set_title('Точность (5 эпох)')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# ==================== ОЦЕНКА МОДЕЛИ ====================
print("\n🧪 Оценка модели на тестовых данных...")

test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)

print(f"📊 РЕЗУЛЬТАТЫ НА ТЕСТОВЫХ ДАННЫХ:")
print(f"✅ Потери (Loss): {test_loss:.4f}")
print(f"✅ Точность (Accuracy): {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Предсказания
y_pred_proba = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# ==================== ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ====================
print("\n📊 Детальный анализ результатов...")

def detailed_analysis(y_true, y_pred, word_dict):
    """Детальный анализ качества классификации"""

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(16, 14))

    # Матрица ошибок
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=[word_dict[i] for i in range(26)],
                yticklabels=[word_dict[i] for i in range(26)])
    plt.title('Матрица ошибок', fontsize=14)
    plt.xlabel('Предсказанные буквы')
    plt.ylabel('Истинные буквы')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Точность по классам
    plt.subplot(2, 2, 2)
    accuracy_per_class = []
    for i in range(26):
        mask = y_true == i
        if mask.sum() > 0:
            accuracy = (y_pred[mask] == i).mean()
            accuracy_per_class.append(accuracy)

    plt.bar(range(26), accuracy_per_class, color='green', alpha=0.7)
    plt.title('Точность по буквам', fontsize=14)
    plt.xlabel('Буква')
    plt.ylabel('Точность')
    plt.xticks(range(26), [word_dict[i] for i in range(26)], rotation=45)
    plt.ylim(0.7, 1.0)
    plt.grid(True, alpha=0.3)

    # Распределение ошибок
    plt.subplot(2, 2, 3)
    errors = y_pred != y_true
    error_distribution = [((y_true[errors] == i).sum()) for i in range(26)]

    plt.bar(range(26), error_distribution, color='red', alpha=0.7)
    plt.title('Распределение ошибок по буквам', fontsize=14)
    plt.xlabel('Буква')
    plt.ylabel('Количество ошибок')
    plt.xticks(range(26), [word_dict[i] for i in range(26)], rotation=45)
    plt.grid(True, alpha=0.3)

    # Сравнение тренировочной и тестовой точности
    plt.subplot(2, 2, 4)
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    categories = ['Тренировочная', 'Тестовая']
    accuracies = [final_train_acc, final_val_acc]
    colors = ['blue', 'orange']

    bars = plt.bar(categories, accuracies, color=colors, alpha=0.7)
    plt.title('Сравнение точности (5 эпох)', fontsize=14)
    plt.ylabel('Точность')
    plt.ylim(0.8, 1.0)

    # Добавляем значения на столбцы
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Отчет классификации
    print("📋 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССИФИКАЦИИ:")
    print(classification_report(y_test, y_pred,
                            target_names=[word_dict[i] for i in range(26)], digits=3))

detailed_analysis(y_test, y_pred, word_dict)

# ==================== ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ ====================
print("\n👁️ Визуализация предсказаний...")

def visualize_predictions(x_original, y_true, y_pred, word_dict, num_examples=12):
    """Визуализация примеров предсказаний"""

    correct_indices = np.where(y_pred == y_true)[0]
    wrong_indices = np.where(y_pred != y_true)[0]

    print(f"✅ Правильных предсказаний: {len(correct_indices)} ({len(correct_indices)/len(y_true)*100:.2f}%)")
    print(f"❌ Ошибочных предсказаний: {len(wrong_indices)} ({len(wrong_indices)/len(y_true)*100:.2f}%)")

    # Визуализация правильных предсказаний
    if len(correct_indices) > 0:
        print("\n✅ ПРИМЕРЫ ПРАВИЛЬНЫХ ПРЕДСКАЗАНИЙ:")
        correct_samples = np.random.choice(correct_indices, min(num_examples, len(correct_indices)), replace=False)

        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
        axes = axes.ravel()

        for i, idx in enumerate(correct_samples):
            img = x_original[idx].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            confidence = np.max(model.predict(x_test[idx:idx+1], verbose=0))
            true_letter = word_dict[y_true[idx]]
            pred_letter = word_dict[y_pred[idx]]

            axes[i].set_title(f'True: {true_letter}, Pred: {pred_letter}\nConf: {confidence:.3f}',
                            color='green', fontweight='bold')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    # Визуализация неправильных предсказаний
    if len(wrong_indices) > 0:
        print("\n❌ ПРИМЕРЫ ОШИБОЧНЫХ ПРЕДСКАЗАНИЙ:")
        wrong_samples = np.random.choice(wrong_indices, min(num_examples, len(wrong_indices)), replace=False)

        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
        axes = axes.ravel()

        for i, idx in enumerate(wrong_samples):
            img = x_original[idx].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            confidence = np.max(model.predict(x_test[idx:idx+1], verbose=0))
            true_letter = word_dict[y_true[idx]]
            pred_letter = word_dict[y_pred[idx]]

            axes[i].set_title(f'True: {true_letter}, Pred: {pred_letter}\nConf: {confidence:.3f}',
                            color='red', fontweight='bold')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

x_test_original = x_test * 255
visualize_predictions(x_test_original, y_test, y_pred, word_dict)

# ==================== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ И ВЫВОДЫ ====================
print("\n" + "="*70)
print("🎉 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ С ОПТИМИЗАЦИЕЙ")
print("="*70)

final_accuracy = test_accuracy * 100
print(f"\n📊 ОСНОВНЫЕ РЕЗУЛЬТАТЫ:")
print(f"   🎯 Точность на тестовых данных: {final_accuracy:.2f}%")
print(f"   ⏱️  Количество эпох обучения: {len(history.history['accuracy'])}")
print(f"   🔧 Оптимизатор: Adam с настройками")
print(f"   ❌ Ошибок: {(y_pred != y_test).sum()} из {len(y_test)}")

# Анализ эффективности оптимизации
train_final_acc = history.history['accuracy'][-1] * 100
overfitting_gap = train_final_acc - final_accuracy

print(f"\n📈 АНАЛИЗ ЭФФЕКТИВНОСТИ ОПТИМИЗАЦИИ:")
print(f"   Тренировочная точность: {train_final_acc:.2f}%")
print(f"   Тестовая точность: {final_accuracy:.2f}%")
print(f"   Разница: {overfitting_gap:.2f}%")

# Анализ скорости обучения
initial_acc = history.history['val_accuracy'][0] * 100
final_acc = history.history['val_accuracy'][-1] * 100
improvement = final_acc - initial_acc

print(f"   Начальная точность: {initial_acc:.2f}%")
print(f"   Конечная точность: {final_acc:.2f}%")
print(f"   Улучшение за 5 эпох: {improvement:.2f}%")

if improvement > 15:
    print("   🚀 Отличная скорость обучения!")
elif improvement > 10:
    print("   ⚡ Хорошая скорость обучения")
else:
    print("   📉 Скорость обучения можно улучшить")

print(f"\n💾 Сохраняем оптимизированную модель...")
model.save('alphabet_recognition_optimized.h5')
print("✅ Оптимизированная модель сохранена!")