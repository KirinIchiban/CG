# Лабораторная работа №2: Камера и освещение (Vulkan)

## 1. Техническое задание (Условие)

### Цель работы
Реализовать интерактивную камеру и модель освещения Блинн-Фонга (Blinn-Phong) с поддержкой нескольких типов источников света (направленный, точечные, прожекторные) с помощью Vulkan API, GLSL и ImGui.

---

### Требования к реализации (Максимальный балл — 5)

1. **Процедурная генерация нормалей:**
   - Для отрисовываемого объекта (Конус из Лабораторной №1) необходимо процедурно рассчитать нормали для каждой вершины.
   - Обновить структуру вершины `Vertex` для передачи нормалей во фрагментный шейдер.

2. **Модель освещения Блинн-Фонга (Blinn-Phong):**
   - **Ambient (Окружающее освещение):** $I_{ambient} = K_a \cdot L_{ambient}$
   - **Diffuse (Диффузное по Ламберту):** $I_{diffuse} = K_d \cdot L_{color} \cdot \max(\vec{N} \cdot \vec{L}, 0)$
   - **Specular (Блик по Блинн-Фонгу):** $I_{specular} = K_s \cdot L_{color} \cdot \max(\vec{N} \cdot \vec{H}, 0)^p$, где $\vec{H} = \frac{\vec{L} + \vec{V}}{\|\vec{L} + \vec{V}\|}$

3. **Затухание (Attenuation) для Point & Spot источников:**
   - Моделирование закона обратных квадратов:
     $$\text{Factor} = \frac{1.0}{1.0 + k_l \cdot d + k_q \cdot d^2} \quad \text{или} \quad \frac{1.0}{d^2}$$

4. **Мягкие края для Spot light (Прожектор):**
   - Расчет интенсивности на основе внутреннего (`innerCutoff`) и внешнего (`outerCutoff`) углов:
     $$\theta = \vec{L}_{dir} \cdot (-\vec{L})$$
     $$\epsilon = \cos(\text{inner}) - \cos(\text{outer})$$
     $$I_{spot} = \text{clamp}\left(\frac{\theta - \cos(\text{outer})}{\epsilon}, 0.0, 1.0\right)$$

5. **Разделение на два Descriptor Set (Set 0 и Set 1):**
   - **Set 0 (Scene Data):**
     - Binding 0: UBO с матрицами `view`, `proj`, позицией камеры `viewPos` и параметрами `ambientLight`.
     - Binding 1: UBO с параметрами `DirectionalLight`.
     - Binding 2: SSBO (Storage Buffer) для массива `PointLight[]`.
     - Binding 3: SSBO (Storage Buffer) для массива `SpotLight[]`.
   - **Set 1 (Object / Material Data):**
     - Binding 0: UBO с матрицей `model` и нормальной матрицей `normalMatrix` ($\text{transpose}(\text{inverse}(M_{3\times3}))$).
     - Binding 1: UBO или Push Constants с параметрами материала ($K_a, K_d, K_s, p$).

6. **Визуализация источников света:**
   - Отображение точечных источников и прожекторов в виде ярких кубов/сфер с фиксированным цветом (без применения освещения Блинн-Фонга).
   - Отображение направления прожектора конусом/линиями.
   - Возможность переключения видимости маркеров в ImGui.

7. **Интерфейс ImGui:**
   - Управление камерой (позиция, target/angles, FOV).
   - Управление материалом объекта (Albedo, Specular Color, Shininess).
   - Динамическое добавление, удаление и редактирование параметров источников света.
