# 📊 Analizador Sintáctico LL(1)

Esta aplicación proporciona una interfaz gráfica interactiva para el análisis sintáctico LL(1) de gramáticas libres de contexto, desarrollada para el curso de Compiladores - UTEC.

## ✨ Características principales

- 🧩 **Análisis de gramáticas LL(1)**: Analiza gramáticas libres de contexto usando parsing predictivo.
- 🔍 **Cálculo automático de FIRST y FOLLOW**: Visualiza los conjuntos FIRST y FOLLOW con codificación por colores.
- 📋 **Tabla de análisis LL(1)**: Genera la tabla de decisión completa para el parser.
- 🔄 **Optimización de gramáticas**:
  - Eliminación de recursividad por izquierda.
  - Factorización por izquierda.
- 🌳 **Visualización de árboles de derivación**: Usa Graphviz (con respaldo en Matplotlib) para representar la estructura sintáctica.
- 🔢 **Análisis paso a paso**: Muestra el estado de la pila, la entrada pendiente y las acciones realizadas en cada paso.
- 💾 **Exportación de resultados**: Descarga el análisis completo y el árbol como archivos.

## 🚀 Instalación

1. Clona este repositorio o descarga los archivos fuente:
   ```bash
   git clone https://github.com/ianskev/ll-parser.git
   cd ll-parser
   ```

2. Crea y activa un entorno virtual de Python:
   ```bash
   python -m venv venv

   # En Windows:
   venv\Scripts\activate

   # En macOS/Linux:
   source venv/bin/activate
   ```

3. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

## 🏁 Ejecución

Para iniciar la aplicación, ejecuta:
```bash
python run_app.py
``` 
La aplicación se abrirá automáticamente en tu navegador web predeterminado.

## 📘 Guía detallada de funcionalidades

### 1️⃣ Ingreso de gramáticas

- ✏️ **Editor de texto**:
  - Escribe la gramática directamente en la interfaz.
  - Usa el teclado virtual para símbolos especiales:
    - `ε` → Cadena vacía (epsilon)
    - `->` → Separador de producciones
    - `|` → Alternativas
    - `id`, `(`, `)` → Terminales y paréntesis
  - Los símbolos se añaden al final del texto.

- 📁 **Subir archivo**:
  - Carga gramáticas desde archivos `.txt`.
  - Formato de ejemplo:
    ```txt
    E -> E + T | T
    T -> T * F | F
    F -> ( E ) | id
    ```

### 2️⃣ Optimización de gramáticas

- 🔄 **Eliminación de recursividad por izquierda**:
  - Transforma reglas del tipo `A -> A α | β` en forma equivalente sin recursión.
  - **Ejemplo**:

    ❌ **Original**: `E -> E + T | T`  
    ✅ **Optimizado**:
    ```txt
    E  -> T E'
    E' -> + T E' | ε
    ```

- 🔀 **Factorización por izquierda**:
  - Agrupa alternativas con prefijos comunes `A -> α β | α γ`.
  - **Ejemplo**:

    ❌ **Original**: `S -> if E then S | if E then S else S`  
    ✅ **Factorizado**:
    ```txt
    S -> if E then S S'
    S' -> else S | ε
    ```

### 3️⃣ Análisis de cadenas

- ✍️ Ingresa una cadena como `id + id * id`.
- 🔍 La aplicación verifica si la cadena pertenece al lenguaje definido.
- ⚙️ El análisis usa el algoritmo predictivo LL(1).

#### 📊 Resultados del análisis

1. **Tabla de Símbolos**:
   - Símbolos no terminales.
   - Conjuntos FIRST y FOLLOW.
   - Codificación por colores.

2. **Análisis de la Cadena**:
   - Estado de la pila en cada paso.
   - Entrada pendiente.
   - Acciones realizadas.
   - ✅ CADENADA ACEPTADA y ❌ CADENA RECHAZADA

3. **Tabla de Análisis LL(1)**:
   - Filas: no terminales.
   - Columnas: terminales.
   - Celdas: producción a aplicar (celdas vacías = error).

### 4️⃣ Visualización del árbol de derivación

- 🌳 Genera un árbol visual de alta calidad
- 🎨 Colores:
  - 🟦 Azul: no terminales.
  - 🟩 Verde: terminales.


*Desarrollado para el curso de Compiladores - UTEC*  
*2025*  
