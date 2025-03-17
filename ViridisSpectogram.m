%%%%%%%%%%%%%% CONSTRUCCIÓN DE IMÁGENES TIEMPO FRECUENCIA %%%%%%%%%%%%%%%%%
clear 
clc
close all

fs = 200; %dato del archivo json

folder_path = 'E:\Trabajos de Grado\Tesis MPaula y Carlos\Dataset\Dataset_reducido\A';  
file_pattern = fullfile(folder_path, '*.mat'); 
files = dir(file_pattern);
nf = length(files); 

% Carpetas de salida
folder_path_1 = 'E:\Trabajos de Grado\Tesis MPaula y Carlos\Espectogramas';


% Crear una figura para las imágenes
hFig = figure('Position', [100, 100, 224, 224], 'Visible', 'off'); % Configura la figura pero la hace invisible

% Representación Tiempo Frecuencia (RTF)
for k = 1:nf
    file_name = fullfile(folder_path, files(k).name);
    signal = load(file_name);
    signal = signal.signal_combined';

    nombre = files(k).name(1:end-4); % Quitar el .wav del nombre
    nombre = strcat(nombre, '.png');
    
     % Espectrograma mel con configuración  (ventana de 32)
     spectrogram(signal, hann(32, 'periodic'), 16, 256, fs, 'yaxis');
    
    % Cambiar a la paleta de colores viridis
    colormap('viridis');
    
    % Ajustar propiedades del eje
    set(gca, 'Visible', 'off'); % Quita los ejes
    colorbar('off'); % Quita la barra de colores
    
    % Ajustar los límites del eje para eliminar espacio blanco
    axis tight; % Ajusta los ejes a los datos
    ax = gca; % Obtener el eje actual
    ax.Position = [0 0 1 1]; % Ajustar la posición del eje para llenar la figura

    file_name = fullfile(folder_path_1, nombre);
    saveas(hFig, file_name); % Guarda la imagen

end

% Cierra la figura al finalizar
close(hFig);