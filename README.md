# K3D student — 3D DIT Distillation (AIJ Contest 2026)

Быстрый студент вместо стадии `lat_dit` Kandinsky3D. По вокселям объекта (`vox_idxi` [N,3], сетка 64³) он за один
проход предсказывает латент [N,16] для каждого вокселя. Латент затем декодирует штатный `lat_vae`.

- **Модель:** MLP 130 → 16 → 16 → 16 (SiLU), 2 640 параметров. Признаки вокселя — занятость 124 соседей в кубе 5³ и
  положение. Студент обучен регрессией на одношаговое предсказание teacher при t=1 (x0c).
- **Инференс:** свои CUDA-ядра (тензорные ядра, fp16 → fp32) запускаются через CUDA driver API, nvcc не нужен.
  `generate_impl` — один вызов C-расширения. Выходной тензор берётся из пула слотов через DLPack.
- **Лидерборд (A100):** v7 — Total 303 113, Time 343 425, Quality 0.8951 (11-е место на 2026-10-13). Текущая
  версия v9 (`solution/`) — та же модель, вызов на T4 на 39% короче (см. «Результаты»).

## Содержание

```
solution/                 сабмит как есть (= k3d_sub_v9_n16g.zip): solution.py + weights/
  solution.py             класс Solution(LatDitGenerator): эталон на PyTorch + CUDA-движок + быстрый путь
  weights/student.pt      веса студента (H16, формат k3d-student-v1)
  weights/*.cubin         ядра для sm_75 / 80 / 86 / 89 / 90 (из CUDA_SRC в solution.py)
  weights/*.so            k3d_fast9 (быстрый путь, CPython abi3) и k3d_shim7 (шим запуска), Linux x86_64
engine/                   исходники C-расширения и шима, build.sh (пересборка .so и .cubin), SHA256SUMS
teacher/                  цели teacher: ноутбук Kaggle K3D-02 + те же исходники (t4compat, calib, gen)
training/                 упаковка данных, признаки, обучение студента
tools/                    make_submission.py, make_test_latents.py, заглушка протокола kandi3d_image23d
tests/                    CPU-тесты (эмулятор ядер, быстрый путь, весь конвейер обучение → сабмит), GPU-тест
bench/                    бенчмарк задержки K3D-11: ноутбук Kaggle T4, скрипт, results/bench11_kernel.json
latents/public_test/      латенты студента для public_test (555 объектов), manifest.csv, info.json
docs/NOTES.md             журнал экспериментов: калибровка teacher, зонд декодера, размеры студента, версии движка
```

Код и веса baseline (3dgen-main, `kandi3d_weights_aijc_kit`) принадлежат организаторам и в репозиторий не входят.

## Быстрый старт

```bash
pip install -r requirements.txt          # torch, numpy (в контейнере организаторов уже есть)
tests/run_all.sh                         # CPU-тесты, ~1 мин; test_gpu.py запускается при наличии CUDA
python tools/make_submission.py --out k3d_submission.zip   # файлы zip побайтно совпадают с solution/
```

Официальный прогон (нужны 3dgen-main и веса организаторов):

```bash
H=3dgen-main/scripts/hackathon
python $H/run_lat_dit_eval.py --cache-dir public_test/public_test --out-dir out/student \
    --submission-dir solution --lat-vae-dir kandi3d_weights_aijc_kit/lat_vae
python $H/run_lat_dit_eval.py --cache-dir public_test/public_test --out-dir out/teacher \
    --lat-dit-dir kandi3d_weights_aijc_kit/lat_dit --lat-vae-dir kandi3d_weights_aijc_kit/lat_vae
python $H/score.py --gt-csv out/teacher/results_summary.csv --pred-csv out/student/results_summary.csv --out-dir out/score
```

В логе харнесса должно быть `Using custom generator: solution.Solution`, а в логе решения — `[k3d] ready: H=16 vm=global path=engine/<mode>/shim+fast`.

Проверка на GPU без харнесса: `python tests/test_gpu.py [--official-repo 3dgen-main]`.

## Решение

### Модель
- **Признаки** (130 на воксель, зависят только от множества вокселей объекта):
  - `occ` — 124 бита занятости соседей в кубе 5×5×5 без центра;
  - `pos` — x/64 и (x − 31.5)/31.5 по трём осям, округлённые до fp16.
  - При обучении центр и масштаб брались из bbox объекта. Все 1208 объектов train_part_001 и все 555 public_test
    центрированы (центр 31.5, полуразмах 31.5), поэтому в решении это константы. На реальных объектах признаки
    совпадают с обучающими бит в бит (`tests/test_pipeline.py`).
- **Сеть:** Linear(130,16) → SiLU → Linear(16,16) → SiLU → Linear(16,16).
- **Цель — x0c**: одношаговое предсказание teacher при t=1 с картинкой, x0c = denorm(z − v_cond(z, t=1)). Потеря — MSE.
  MSE-цель даёт условное среднее, и декодеру этого хватает: усреднение двух сидов teacher не ухудшает LPIPS (`docs/NOTES.md`).
- **Подгонка разброса** («global»): выход растягивается по каналам вокруг среднего, с коэффициентом
  std(цели) / std(предсказания) на обучающей выборке. Коэффициент вшит в последний слой. Без этого латент слишком
  гладкий, а сглаживание декодер не прощает.
- **Признаки DINOv3 не используются.** Картинка мало меняет одношаговое предсказание: медиана rel MSE(x0u, x0c) 0.06.
  Teacher с картинкой выигрывает у teacher без неё лишь ~0.01 LPIPS. Признаки картинки с токеном DINO переобучались.
- Размер H = 16 выбран по итоговому баллу. Вызов почти целиком состоит из постоянных накладных, поэтому меньшая сеть
  выигрывает во времени больше, чем теряет в качестве. Переход v6 H64 → v7 H16 на лидерборде: Quality −1.4%,
  Time +12.4%, Total +10.6%.

### Инференс (`solution/solution.py`)
- Два пути одной функции:
  - `_Reference` — PyTorch на любом устройстве. Это эталон для самопроверок и запасной путь.
  - `_Engine` — CUDA-ядра движка v7 через driver API (`libcuda`), без nvcc в рантайме.
    1. `stamp` строит битовую сетку занятости (64-битная строка на (x, y)) и обнуляет буфер следующего вызова.
    2. Ядро MLP собирает признаки из сетки и прогоняет сеть на тензорных ядрах (mma m16n8k16, fp16 → fp32) тайлами по
       16 вокселей. Веса читаются из shared memory, активации остаются в регистрах.
  - Режимы запуска: split (два ядра), fused (одно ядро с барьером по сетке), coop (кооперативный запуск).
- **Загрузка** (`from_pretrained`, вне замера):
  1. Чтение весов и упаковка во фрагменты.
  2. Загрузка cubin для своей архитектуры. Если подходящего нет — компиляция NVRTC.
  3. Самопроверка каждого режима против эталона на синтетических фигурах (сферы, слои, случайные точки, углы сетки,
     обе раскладки входа).
  4. Выбор режима по замеру с холодными кэшами.
  5. Проверка быстрого пути тем же вызовом, что делает харнесс.
  - Любой сбой понижает путь: быстрый путь → Python + шим → ctypes → PyTorch.
- **Вызов** (`generate_impl` = `k3d_fast9`, один C-вызов):
  1. Проверки входа (int32, CUDA, [N,3], любые страйды — харнесс передаёт колонко-мажорный тензор).
  2. Слот выхода и запуск ядра.
  3. Только потом сборка объекта тензора (`torch._C._from_dlpack`): GPU считает, пока CPU работает.
  - Слот занят, пока жива память тензора (deleter DLPack). Если свободного слота нет, выход выделяется через
    `new_empty`. Кривые входы уходят в Python-путь без изменений.
- **Все вычисления, зависящие от входа, идут внутри `generate_impl`** при каждом вызове: сетка, признаки, сеть.
  Между вызовами ничего не кэшируется. `from_pretrained` тестовых данных не видит.
- Переменные окружения для диагностики: `K3D_ENGINE` = auto | split | fused | coop | off, `K3D_FAST` = off,
  `K3D_FLAGS`, `K3D_VERBOSE` = 0. Описаны в docstring `solution.py`.

### Совместимость
- Нужны Linux, драйвер NVIDIA (`libcuda.so.1`), torch и numpy. GPU-путь проверен на Kaggle T4 (torch 2.11) и на
  лидерборде (A100). CPU-путь и тесты — torch 2.6, x86_64 и aarch64.
- Ядра: sm_75 и новее (T4, A100, A10/30xx, L4/40xx, H100). Для других архитектур NVRTC компилирует ядра при загрузке,
  если доступен `libnvrtc` (пакет `nvidia-cuda-nvrtc-cu12`, его ставит torch).
- `.so` собраны под x86_64. На других CPU ядра запускаются через ctypes из Python — медленнее, но с тем же результатом.

## Воспроизведение обучения

Цепочка: цели teacher → упаковка данных → обучение → сабмит. Студент обучен на одном train-чанке (`train_part_001`,
1208 объектов, 12.4 млн вокселей).

**1. Цели teacher** (`teacher/`). Нужен GPU и baseline организаторов.

- Kaggle: `teacher/k3d_02_teacher_targets.ipynb`, Accelerator GPU T4 ×2, Internet On, запуск коммитом (Save & Run All).
  - Ноутбук сам качает код, веса и чанк с хранилища организаторов (`teacher/calib/k3d_data.py`) и собирает CUDA-опы.
  - **Фаза 1** (~1.2 ч): для каждого объекта два прохода teacher при t=1, с картинкой и без. Результат — файлы
    `targets/x0/<id>.npz` с полями x0c, x0u, z, все [N,16] fp16. Шум z: seed 42, генератор CUDA.
  - **Фаза 2** (8 шагов, `targets/lat8`) студенту не нужна. Её можно выключить: `PHASE2 = False` в ячейке 0.
  - Результат — `targets.zip` в Output.
- Своя машина с CUDA и локальными данными:

  ```bash
  for d in train_part_001/*/; do touch "$d/.ready"; done; touch train_part_001/.done
  python teacher/gen/run_gen.py --repo 3dgen-main --compat teacher --weights kandi3d_weights_aijc_kit \
      --watch train_part_001 --out targets --no-phase2        # несколько GPU: --shard k --nshards K
  ```

  `teacher/t4compat` подменяет то, что не работает на T4 и CPU: flash-attn → SDPA, bf16 → fp16, CUDA-опы → PyTorch.
  На A100 с установленным flash-attn он ничего не трогает.

**2. Данные и обучение** (CPU, 4 потока).

```bash
python training/prep_data.py  --inputs train_part_001 --targets targets --out data/c001  # memmap, ~1.8 ГБ
python training/prep_train.py --data data/c001        # train: 3.0 млн вокселей (1089 объектов), val: 1.32 млн (119)
python training/train_x0c.py  --data data/c001 --run runs/h16      # 60k шагов × 4096 вокселей, ~3–4 мин
python tools/make_submission.py --student runs/h16/student.pt --out k3d_submission.zip
```

- Все параметры по умолчанию соответствуют поставленному студенту. В конце `train_x0c.py` печатает val rel MSE по
  объектам. У поставленного студента 0.5805.
- Обучение детерминировано (`--seed 0`) и не зависит от того, на сколько вызовов разбит прогон (`--budget`).
  Побитово исходные веса не повторить: исходный прогон шёл без сида.
- Проверено с нуля (`docs/NOTES.md`):
  - данные из целей teacher получаются побайтно те же;
  - повторное обучение даёт val 0.5792 против 0.5805 у поставленного студента.

## Воспроизведение инференса

- `engine/build.sh` пересобирает `k3d_fast9_x86_64.abi3.so`, `k3d_shim7_x86_64.so` и cubin в `solution/weights`.
  С gcc 13.3, заголовками Python 3.11 и NVRTC 12.4.127 файлы побайтно совпадают с закоммиченными (`engine/SHA256SUMS`).
- Ядра — строка `CUDA_SRC` в `solution.py`. Тег cubin — первые 12 hex-символов sha256 этой строки.
  `solution.py` берёт только cubin своего тега.
- `tools/make_submission.py` проверяет:
  - теги cubin;
  - имена расширений;
  - раскладку zip (официальной `inspect_zip`, если передан `--official-repo`);
  - что студент из распакованного zip загружается и считает.

## Латенты для теста

```bash
python tools/make_test_latents.py --data public_test/public_test --out latents/public_test [--device cuda] [--check]
```

- Для каждого объекта: `latents/public_test/<id>.npz`, поле `latent` [N,16] float16. Строки идут в порядке
  `vox_idxi` — это ровно то, что `generate()` отдаёт декодеру `lat_vae`.
- `manifest.csv`: id, N, dtype, sha256 массива. `info.json`: устройство, путь, md5 `solution.py` и `student.pt`.
- Вызов тот же, что у харнесса: `torch.tensor(vox_idxi, int32)` → `Solution.generate_impl`.
  - На CUDA работает движок. Он совпадает с эталоном до rel MSE ~4e-6 (K3D-11, T4).
  - На CPU работает эталон PyTorch.
  - Поставленные латенты посчитаны на CPU (эталонный путь). `--dtype float32` сохраняет без округления.
- 555 объектов, 5.68 млн вокселей, 161 МБ. На CPU x86_64 и aarch64 латенты совпадают, кроме 2 чисел из 90.9 млн
  (1 ulp fp16).

## Результаты

**Лидерборд (A100, public_test).** Total = среднее quality × T_teacher / T_student.

| Дата | Версия | Total | Time | Quality |
|---|---|---|---|---|
| 2026-10-11 | v4, H64 — веса в shared, режимы split/fused/coop | 168 105 | 187 574 | 0.9076 |
| 2026-10-12 | v6, H64 — C-расширение, вход по страйдам без копии | 274 118 | 305 612 | 0.90757 |
| 2026-10-13 | v7, H16 — вдвое меньше инструкций на тайл | 303 113 | 343 425 | 0.8951 |

**Задержка вызова на T4** (K3D-11, `bench/`), медиана по объектам public_test, мкс:

| | подряд | цикл как в харнессе | после 1 с простоя |
|---|---|---|---|
| генератор, который только выделяет выход (`new_empty`) | 15.6 | 42.2 | 39.9 |
| v6 H64 | 35.6 | 81.3 | 96.3 |
| v7 H16 | 28.8 | 76.5 | 86.6 |
| **v9 H16** | **21.0** | **46.7** | **50.2** |

- «Цикл как в харнессе» имитирует официальный прогон: между вызовами работает декодер, CPU делает постобработку,
  входы копируются на GPU.
- В таком режиме вызов почти целиком состоит из постоянных накладных CPU: генератор без вычислений стоит 42 мкс.
  Поэтому v9 убрал из пути выделение памяти через диспетчер PyTorch и запускает ядро раньше. Весь вызов v9 дороже
  такого генератора лишь на 4.5 мкс.

**Качество от размера студента:**

| H | val rel MSE x0c | Quality (A100) |
|---|---|---|
| 16 | 0.581 | 0.8951 |
| 32 | 0.524 | — |
| 48 | 0.497 | — |
| 64 | 0.481 | 0.9076 |
| 128 | 0.448 | — |

Подробности экспериментов — в `docs/NOTES.md`.

## Лицензия
MIT (`LICENSE`) — на код и веса этого репозитория.
