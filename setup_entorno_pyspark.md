# Configuración del entorno local — Proyecto MIND RecSys

1. Se creó la carpeta del proyecto en `C:\DevProjects\MMDS\RecSysMIND`, sin espacios ni tildes en la ruta.
2. Se verificó la versión de Python del sistema (3.14) y se determinó que era demasiado reciente para PySpark, por lo que se optó por usar Python 3.11 para el proyecto.
3. Se instaló y configuró el Python Install Manager (reemplazo oficial del antiguo `py.exe`), habilitando los alias de ejecución de aplicaciones en Windows.
4. Se confirmó que Python 3.11 ya estaba disponible en el sistema mediante `py list`, evitando una instalación redundante.
5. Se creó un entorno virtual del proyecto con `py -3.11 -m venv venv`, aislado del Python 3.14 del sistema.
6. Se activó el entorno virtual (`.\venv\Scripts\Activate.ps1`) y se confirmó la versión correcta con `python --version`.
7. Se instaló PySpark (versión 4.0.2) junto con Jupyter, pandas y matplotlib mediante `pip install`.
8. Se instaló Java Development Kit 17 (Eclipse Temurin), requerido por Spark 4.x, marcando las opciones de `JAVA_HOME` y `PATH` durante la instalación.
9. Se verificó la instalación de Java con `java --version` y se confirmó la variable `JAVA_HOME`.
10. Se descargaron `winutils.exe` y `hadoop.dll` (versión Hadoop 3.3.6, la más reciente disponible) desde el repositorio `cdarlint/winutils`, necesarios para que Spark funcione en Windows.
11. Se ubicaron ambos archivos en `C:\hadoop\bin` y se configuró la variable de entorno `HADOOP_HOME=C:\hadoop`, agregando `%HADOOP_HOME%\bin` al PATH.
12. Se probó el arranque de una `SparkSession` local desde Python, confirmando la versión de Spark (4.0.2) y ejecutando una consulta de prueba (`spark.range(5).show()`) sin errores.
