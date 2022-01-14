## PAR Cuda

### Instalación

#### Preparación

Para configurar el entorno de trabajo es necesario la instalación de las siguientes herramientas.

- cmake
- build-essential
- git (para clonar el repositorio)

Estas herramientas pueden ser instaladas de la siguiente manera (Ubuntu): 
```
sudo apt install build-essential cmake
```

#### Clonar el repositorio

```
git clone https://github.com/Asiern/parcuda.git
```

#### Configuración

```
cmake -B build -DUSE_DEBUG=[ON/OFF]
```

### Compilación

Una vez configurado el entorno de trabajo, el proyecto se puede compilar con el siguiente comando.
```
cd build && make
```

### Estructura del código fuente

| Directorio | Descripción                                                                                                                      |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- |
| /src       | Contiene el código fuente                                                                                                        |
| /lib       | Librerias                                                                                                                        |
| /build     | Directorio destino para los archivos compilados (Esta carpeta se crea automáticamente cuando se configura el entorno de trabajo) |
