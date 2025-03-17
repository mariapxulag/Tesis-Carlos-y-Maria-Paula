
clear 
clc
close all

folder_path = 'E:\Trabajos de Grado\Tesis MPaula y Carlos\Dataset\Dataset_full'; %se debe modificar según cada 
file_pattern = fullfile(folder_path, '*.json'); 
files = dir(file_pattern);
nf = length(files); %número de señales 

%% Extraer datos
% Variable para almacenar todos los datos de EMG
all_emg_data = zeros(400,8);
% Variable para almacenar todos los datos de IMU
all_imu_data = zeros(400,10);

% Leer y procesar cada archivo
for i = 1:nf
    acceleration_data = [];
    gyroscope_data = [];
    orientation_data = [];
    % Leer el archivo JSON
    file_name = fullfile(folder_path, files(i).name);
    rawData = jsondecode(fileread(file_name));
    
    % Extraer datos de EMG
    emgData = rawData.emg.data;
    
    % Concatenar datos de EMG de todos los archivos
    all_emg_data = all_emg_data + emgData;

    % Extraer datos de IMU (si está presente)
   % if isfield(rawData, 'imu') && isfield(rawData.imu, 'data')
        imuData = rawData.imu.data;
         for j = 1:length(imuData)
            % Extraer atributos de cada entrada y concatenar
            acceleration_data = [acceleration_data; imuData(j).acceleration'];
            gyroscope_data = [gyroscope_data; imuData(j).gyroscope'];
            orientation_data = [orientation_data; imuData(j).orientation'];
        end
        all_imu_data = all_imu_data +  [acceleration_data gyroscope_data orientation_data];
   % end
end
normalized_emg = (all_emg_data - mean(all_emg_data, 1)) ./ std(all_emg_data, 0, 1);
normalized_imu = (all_imu_data - mean(all_imu_data, 1)) ./ std(all_imu_data, 0, 1);
normalized_data = [normalized_emg normalized_imu];


%% PCA en datos EMG e IMU
% Aplicar PCA
    [coeff, score, latent, tsquared, explained] = pca(normalized_data);
    
    % Número de componentes principales a retener (porcentaje de varianza explicada > 95%)
    cumulative_variance = cumsum(explained);
    num_components = find(cumulative_variance > 95, 1);
    
    % Reducir dimensionalidad
    reduced_data = score(:, 1:num_components);
    
    % Visualizar porcentaje de varianza explicada
    figure;
    plot(cumulative_variance, 'LineWidth', 2);
    xlabel('Número de Componentes Principales');
    ylabel('Varianza Explicada Acumulativa (%)');
    title('Análisis de PCA');
    grid minor;
    set(gcf,'Color','white')

    % Guardar los datos reducidos para construir espectrogramas
    save('reduced_data.mat', 'reduced_data', 'coeff', 'explained');

%CAMINOS PARA ANALIZAR SI ERA MEJOR APLICAR PCA DE FORMA INDEPENDIENTE O CONJUNTA 
%PERO A MI PARECER ES MEJOR HACERLO DE FORMA CONJUNTA 
%POR ESO DEJO ESTO COMENTADO

% %% PCA en datos EMG
% if ~isempty(all_emg_data)
% 
%     % Aplicar PCA
%     [coeff_emg, score_emg, latent_emg, tsquared_emg, explained_emg] = pca(normalized_emg);
% 
%     % Número de componentes principales (95% varianza explicada)
%     cumulative_variance_emg = cumsum(explained_emg);
%     num_components_emg = find(cumulative_variance_emg > 95, 1);
%     reduced_emg_data = score_emg(:, 1:num_components_emg);
% 
%     % Visualizar varianza explicada
%     figure;
%     plot(cumulative_variance_emg, 'LineWidth', 2);
%     xlabel('Número de Componentes Principales (EMG)');
%     ylabel('Varianza Explicada Acumulativa (%)');
%     title('PCA para datos EMG');
%     grid minor;
%     set(gcf,'Color','white')
% 
%     % Guardar datos EMG reducidos
%     save('reduced_emg_data.mat', 'reduced_emg_data', 'coeff_emg', 'explained_emg');
% end
% 
% %% PCA en datos IMU
% if ~isempty(normalized_imu)
%     % Aplicar PCA
%     [coeff_imu, score_imu, latent_imu, tsquared_imu, explained_imu] = pca(normalized_imu);
% 
%     % Número de componentes principales (95% varianza explicada)
%     cumulative_variance_imu = cumsum(explained_imu);
%     num_components_imu = find(cumulative_variance_imu > 95, 1);
%     reduced_imu_data = score_imu(:, 1:num_components_imu);
% 
%     % Visualizar varianza explicada
%     figure;
%     plot(cumulative_variance_imu, 'LineWidth', 2);
%     xlabel('Número de Componentes Principales (IMU)');
%     ylabel('Varianza Explicada Acumulativa (%)');
%     title('PCA para datos IMU');
%     grid minor;
%     set(gcf,'Color','white')
% 
%     % Guardar datos IMU reducidos
%     save('reduced_imu_data.mat', 'reduced_imu_data', 'coeff_imu', 'explained_imu');
% end

%% Crear una única señal para los diferentes casos
folder_path_load = 'E:\Trabajos de Grado\Tesis MPaula y Carlos\Dataset\Dataset\A'; %se debe modificar según cada equipo y según la letra
folder_path_save = 'E:\Trabajos de Grado\Tesis MPaula y Carlos\Dataset\Dataset_reducido\A'; %se debe modificar según cada equipo y según la letra
file_pattern = fullfile(folder_path_load, '*.json'); 
files = dir(file_pattern);
nf = length(files); %número de señales 


ncp = 18; %número de componentes principales (todos)
          %se puede dejar en 9 que sería el úmero de componentes principales en los que se encuentra el 95% de la información 

% Variable para almacenar todos los datos 
all_data = zeros(400,18);


% Leer y procesar cada archivo
for i = 1:nf
    acceleration_data = [];
    gyroscope_data = [];
    orientation_data = [];
    % Leer el archivo JSON
    file_name = fullfile(folder_path, files(i).name);
    rawData = jsondecode(fileread(file_name));
    
    % Extraer datos de EMG
    emgData = rawData.emg.data;
    
    % Extraer datos de IMU (si está presente)
   % if isfield(rawData, 'imu') && isfield(rawData.imu, 'data')
        imuData = rawData.imu.data;
         for j = 1:length(imuData)
            % Extraer atributos de cada entrada y concatenar
            acceleration_data = [acceleration_data; imuData(j).acceleration'];
            gyroscope_data = [gyroscope_data; imuData(j).gyroscope'];
            orientation_data = [orientation_data; imuData(j).orientation'];
        end
    % end
   all_data  =  [emgData acceleration_data gyroscope_data orientation_data];
   normalized = (all_data - mean(all_data, 1)) ./ std(all_data, 0, 1);
   % Seleccionar los scores de los componentes principales
   score_new = normalized * coeff;
   % Crear la señal única combinando las componentes principales
   % Usamos una suma ponderada de los scores
   signal_combined = sum(score_new, 2);
   % (Opcional) Normalizar la señal combinada
   signal_combined = (signal_combined - mean(signal_combined)) / std(signal_combined);
   %guardamos la nueva señal
   filename = [files(i).name(1:end-5) '.mat'];
   file_path = fullfile(folder_path_save, filename);
   save(file_path, 'signal_combined');
end

