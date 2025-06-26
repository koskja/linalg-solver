# Zápočtový projekt: Knihovna pro lineární algebru s LaTeX výstupem

## 1. Zadání problému

Cílem projektu je vytvořit knihovnu v jazyce Python, která umožňuje provádět základní výpočty lineární algebry a zároveň generovat podrobný, lidsky čitelný zápis celého postupu ve formátu LaTeX. 

Knihovna by měla:
- Poskytovat funkce pro elementární operace (sčítání, násobení, transpozice)
- Poskytovat funkce pro netriviální maticové operace (determinant, inverze, řešení SLR)
- Poskytovat funkce pro práci s vlastními čísly (určení vlastních čísel a vektorů, diagonalizace)
- Kromě samotného výsledku také vrátit postup výpočtu
- Být schopna generovat náhodné matice se zvolenými vlastnostmi

Posledním (a nejdůležitějším) požadavkem je, aby bylo možné poskytované postupy smysluplně komponovat a vytvářet tak komplexnější výpočty.

## 2. Použití

### Hlavní principy práce s knihovnou

Knihovna je navržena kolem centrální třídy `Matrix`, která reprezentuje matice a poskytuje metody pro lineárně-algebraické operace. Klíčovou vlastností knihovny je automatické generování LaTeX dokumentace pro každý výpočetní krok.

**Základní workflow:**
1. **Deklarace matice** – předání dvourozměrného seznamu hodnot konstruktoru `Matrix`.
2. **Výpočet** – voláním metod (např. `determinant`, `inverse`, `transpose` …) proběhne požadovaná operace; každá z nich zároveň průběžně ukládá LaTeX-ový log celého postupu.
3. **Export výsledků** – metoda vrací numerický či symbolický výsledek a současně máte k dispozici kumulovaný LaTeX zápis výpočtu (který lze přímo získat např. použitím fce. `capture_logs`).

**Logging systém:**
- Výstup se shromažďuje v objektech `Logger`, které jsou uloženy na zásobníku (`push_logger`/`pop_logger`). Funkce `log(message, *args)` vždy zapisuje do aktuálně nejvyššího loggeru a zprávu předem LaTeX-ově naformátuje přes `pcformat`.
- Implicitně existuje globální logger. Nastavením `Logger._auto_print = True` (nebo parametrem konstruktoru) se všechny příchozí zprávy současně vypisují na standardní výstup.
- K dočasnému odklonění výstupu použijte kontextové manažery:
  ```python
  with nest_logger() as lg:          # vytvoří nový izolovaný logger
      A.determinant()
  latex_code = str(lg)               # kompletní LaTeX záznam průběhu výpočtu
  ```
- Varianta `nest_appending_logger(accum)` směruje výstup do nového loggeru a po jeho uzavření vloží výsledek do seznamu `accum`. Hodí se pro sběr logů z vícero zanořených výpočtů.
- Funkce `capture_logs(f)` spustí libovolnou funkci `f` s vlastním loggerem a vrátí kumulovaný záznam jako jeden řetězec.

### Dostupné funkce

#### Základní maticové operace
- `Matrix(items: List[List])` - konstruktor matice z dvourozměrného seznamu
  - **Validace**: 
    - Vnější seznam nesmí být prázdný
    - Všechny prvky musí být seznamy (řádky)
    - Všechny řádky musí mít stejnou nenulovou délku

- sčítání, odčítání, násobení matic, násobení matice skalárem
- `matrix.transpose()` - transpozice matice

#### Pokročilé maticové operace
- `matrix.determinant()` - determinant matice
- `matrix.inverse()` - inverzní matice
  - **Vrací**: `Matrix.NoSolution()` pro singulární matice
  
- `matrix.rank()` - hodnost matice
  
- `matrix.row_reduce(bar_col: int | None)` - Gaussova eliminace matice s možností označit rozšířený sloupec. Používá se hlavně interně, ale lze ji použít k vlastním účelům.
  - **Vrací**: `tuple` obsahující:
    - `reduced_items: List[List[Any]]`: redukovaná matice jako seznam seznamů
    - `pivots: List[Tuple[int, int]]`: pozice pivotních prvků
    - `intermediate_matrices: List[str]`: LaTeX řetězce mezikroků
    - `intermediate_steps: List[Tuple[str, str]]`: popis jednotlivých kroků

  **Parametr `bar_col`**: Určuje pozici svislé čáry v rozšířené matici. 
  - Pokud není zadán, automaticky se nastaví na poslední sloupec (`n-1`)
  - Eliminace probíhá pouze do sloupce `bar_col` (exclusive), pravá strana se pouze transformuje
  - Případ `n-1` se používá při řešení soustav lineárních rovnic ve tvaru `[A|b]`, kde matice `[A|b]` ma n sloupcu
  - Jiné hodnoty lze použít pro řešení problémů zahrnujících inverzi matice, např. pro eliminaci rozšířených matic typu `[A | I]`, kde A a I mají stejný řád

#### Lineární systémy
- `matrix.find_preimage_of(vec: List)` - řešení lineárního systému
  - **Vrací**: `Matrix.AffineSubspace` nebo `Matrix.NoSolution()`
  
  **AffineSubspace** - reprezentuje afinní podprostor řešení ve tvaru `vec + LO {generators_matrix[0], generators_matrix[1], ...}`:
  - **Metody**:
    - `.get_one()` → partikulární řešení soustavy
    - `.dim()` → dimenze prostoru řešení (= počet volných proměnných)
    - `.basis()` → báze homogenního prostoru řešení (sloupcové vektory)
- `matrix.kernel()` - jádro (nullspace) matice. Též vrací `Matrix.AffineSubspace`.

#### Vlastní čísla a vektory
- `matrix.eigenvalues(real_only=False)`
  - **Vrací**: `Dict[eigenvalue, multiplicity]`

- `matrix.eigenvalues_with_geometric_multiplicities()`
  - **Vrací**: `Dict[eigenvalue, (alg_mult, geom_mult)]`

- `matrix.find_eigenspace(eigenvalue)` - vlastní podprostor pro dané vlastní číslo
  
- `matrix.diagonalize()` - diagonalizace matice
  - **Vrací**: `DiagonalizationResult` s atributy `.success`, `.P`, `.P_inv`, `.D`. Pokud byla diagonalizována matice `A`, platí `A = P * D * P_inv`

#### Generování náhodných matic

**RandomMatrixBuilder** - flexibilní nástroj pro vytváření matic se specifickými vlastnostmi:

**Základní konfigurace:**
- `RandomMatrixBuilder.new()` - vytvoří nový builder
- `.with_size(rows, cols)` - nastaví rozměry matice
- `.with_dist(distribution_function)` - nastaví rozložení hodnot (default: `random.randint(-5, 5)`)
- `.build()` - vytvoří matici podle specifikace

**Specialní vlastnosti (vzájemně se vylučují):**
- `.with_rank(rank)` - matice s danou hodností
- `.with_eigenvalues(eigenvalues)` - diagonalizovatelná matice s danými vlastními čísly
- `.with_jordan_blocks(blocks)` - matice s Jordanovými bloky

**Pomocné funkce (zkratky pro časté případy):**
- `gen_regular_matrix(N, dist=None)` - regulární N×N matice
- `gen_matrix_with_rank(rows, cols, rank, dist=None)` - matice s danou hodností  
- `gen_diagonalizable_matrix(N, eigenvalues=None, dist=None)` - diagonalizovatelná matice
- `gen_matrix_with_jordan_blocks(N, blocks, dist=None)` - matice s Jordanovými bloky

#### Utility funkce
- `Matrix.zero(rows, cols)` - nulová matice
- `Matrix.identity(size)` - jednotková matice
- `Matrix.diagonal(items)` - diagonální matice
- `Matrix.new_vector(items)` - sloupcový vektor

### Příklady použití

#### Základní operace

```python
A = Matrix([[2, 1], [1, 3]])
log(r"Determinant: %s", A.determinant())
log(r"Hodnost: %s", A.rank())
log(r"Transpozice: %s", A.transpose())

# Řešení Ax = b
solution = A.find_preimage_of([5, 7])
log(r"Řešení: %s", solution)
```

#### Vlastní čísla a diagonalizace

```python
A = Matrix([[4, -2], [-2, 1]])
eigenvals = A.eigenvalues()
result = A.diagonalize()
log(r"Vlastní čísla: %s", eigenvals)
log(r"Diagonalizovatelná: %s", result.success)
if result.success:
    log(r"%s", result)
```

#### Generování náhodných matic

```python
from linalg_solver.random_matrix import RandomMatrixBuilder

# Matice s hodností 2
M = RandomMatrixBuilder.new().with_size(4, 4).with_rank(2).build()
log(r"Hodnost: %s", M.rank())

# Diagonalizovatelná matice s vlastními čísly [1, 2, 3]
D = RandomMatrixBuilder.new().with_size(3, 3).with_eigenvalues([1, 2, 3]).build()
log(r"Vlastní čísla: %s", D.eigenvalues())

# Jordan bloky
J = RandomMatrixBuilder.new().with_size(4, 4).with_jordan_blocks([(1, 2), (0, 2)]).build()
log(r"Matice: %s", J)
```

## 3. Programátorská část

### Architektura

**`linalg_solver/linalg.py`** - Hlavní třída `Matrix` a všechny maticové operace  
**`linalg_solver/fmt.py`** - LaTeX formátování pomocí `cformat()` a raw strings  
**`linalg_solver/log.py`** - Stack-based logging pro vnořené výpočty  
**`linalg_solver/random_matrix.py`** - `RandomMatrixBuilder` pro generování testovacích matic  
**`linalg_solver/permutation.py`** - Permutace pro determinanty  

### Klíčové principy

#### No-log fallback
Pokud není vyžadován postup, funkce se odkáže na implementaci v SymPy, která pravděpodobně bude optimalizovanější.

#### Hierarchický logging
Stack-based logging umožňuje vnořené výpočty - každá operace může volat podoperace a jejich logy se automaticky organizují. Při volání `log()` se záznam přidá do aktuálního "vrchního" loggeru na zásobníku.

```python
log("počítám a počítám")
logs = []
with nest_appending_logger(logs):
    result = some_operation()
# Zbytek výpočtu
log("stále počítám")
# Postup mezivýpočtu se zaznamená později, 
# aby samotný výpočet byl v jednom kuse
log('\n'.join(logs))
```

#### Type-safe results
V případě, že neexistuje řešení, funkce vrátí hodnotu reprezentující chybu. To umožňuje přímočařejší control flow než výjimky.
```python
result = matrix.find_preimage_of([1, 2])
if isinstance(result, Matrix.NoSolution):
    log(r"Žádné řešení")
else:
    log(r"Řešení: %s", result.get_one())
```

### Algoritmy

- **Řešení SLR, inverze**: Klasická Gaussovská eliminace. Interně používá `row_reduce`.
- **Determinant**: Explicitní suma přes všechny permutace
- **Vlastní čísla**: Charakteristický polynom se spočítá pomocí determinantu. Kořeny se naleznou pomocí SymPy. Algebraické násobnosti jsou zřejmé, pro geometrické násobnosti a příslušné vlastní prostory se vyřeší vhodná homogenní soustava lineárních rovnic.
- **Generování matic**: 
  - Základní: náhodné prvky podle distribuce
  - S plnou hodností: opakovaná náhodná volba základní matice. Singulární matice jsou vzácné.
  - S hodností r: násobení A(m×r) × B(r×n) kde obě mají plnou hodnost
  - Diagonalizovatelné: diagonální matice s danými vlastními čísly, pak P⁻¹DP
  - Jordanovy bloky: Normální Jordanova forma, pak podobnostní transformace P⁻¹JP

## 4. Přání do budoucna
 - Ve vhodných situacích determinanty počítat pomocí Gaussovy eliminace. Také:
   - Dokázat provést rozvoj podle sloupce/řádku.
   - Vybrat postup podle toho, aby jeho LaTeX output byl v nějakém smyslu minimální
 - Při generování matic se nabízí spousta možností. Vyčtu zde některé vlastnosti, které nejsou implementovány:
   - Symetrie / antisymetrie (hermitovskost / antihermitovskost)
   - Horní / dolní trojúhelníkovost
   - Normálnost, unitarita
 - Algoritmy zahrnující skalární součin
