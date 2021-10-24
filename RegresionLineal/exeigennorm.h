#ifndef EXEIGENNORM_H
#define EXEIGENNORM_H

#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

class ExEigenNorm
{

     // Para el constructor se necesitarán 3 argumentos de entrada (atributos).


     // 1. Nombre del dataset.

     std::string setDatos;

      //* 2. Separador de columnas.

     std::string delimitador;

     //3. Tiene o no cabecera.

     bool header;

public:
    ExEigenNorm(std::string Datos, std::string separador, bool head):
        setDatos(Datos),
        delimitador(separador),
        header(head){}

    std::vector<std::vector<std::string>> LeerCSV();
    Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> datosString, int filas, int col);
    auto Promedio(Eigen::MatrixXd Datos) ->decltype(Datos.colwise().mean());
    auto Desviacion(Eigen::MatrixXd Datos) -> decltype(((Datos.array().square().colwise().sum())/(Datos.rows()-1)).sqrt());
    Eigen::MatrixXd Normalizacion(Eigen::MatrixXd Datos);
    std::tuple <Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd Datos,float sizeTrain);
    void VectorToFile(std::vector<float> vector, std::string nombre);
    void EigenToFile(Eigen::MatrixXd datos, std::string nombre);
};

#endif // EXEIGENNORM_H
