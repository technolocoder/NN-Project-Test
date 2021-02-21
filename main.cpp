#include <SDL2/SDL.h>

#include <iostream>

#include<vector>

#include<random>

#include <algorithm>

#include <functional>

enum ACTIVATION_FUNCTIONS {TANH,SIGMOID,RELU,LEAKY_RELU,LINEAR};

std::random_device rd;

std::mt19937_64 engine(rd());

template<typename T> void print_vector(const std::vector<T> &input){

    for(int i = 0; i < input.size(); ++i){

        std::cout << input[i] << ' ';

    }

    std::cout << '\n';

}

template<typename T> class neural_network_vector{

public:

    typedef std::vector<T> vector;

    typedef std::vector<vector> matrix;

    neural_network_vector(){}

    neural_network_vector(const int input_size ,const std::vector<int> layer_size, const std::vector<ACTIVATION_FUNCTIONS> layer_activations){

        layers.reserve(layer_size.size()+1);

        layers.emplace_back(input_size);

        layers.insert(layers.end(),layer_size.begin(),layer_size.end());

        activations = layer_activations;

        weights = std::vector<matrix>(layer_size.size());

        bias = matrix(layer_size.size());

        std::uniform_real_distribution<T> dist(-1,1);

        auto generator = std::bind(dist,std::ref(engine));

        for(int i = 0; i < layer_size.size(); ++i){

            weights[i] = matrix(layer_size[i],vector(layers[i]));

            for(int j = 0; j < layer_size[i]; ++j) std::generate(weights[i][j].begin(),weights[i][j].end(),generator);

            bias[i] = vector(layer_size[i]);

            std::generate(bias[i].begin(),bias[i].end(),generator);

        }

    }

    void print_parameters(){

        std::cout << "Weights:\n";

        for(int i = 1; i <= weights.size(); ++i){

            std::cout << "layer no. " << i << ":\n";

            for(int j = 0; j < layers[i]; ++j){

                for(int k = 0; k < layers[i-1]; ++k) std::cout << weights[i-1][j][k] << ' ';

                std::cout << '\n';

            }

            std::cout << '\n';

        }

        std::cout << "Bias:\n";

        for(int i = 0; i < weights.size(); ++i){

            std::cout << "layer no. " << i+1 << ":\n";

            for(int j = 0; j < bias[i].size(); ++j){

                std::cout << bias[i][j] << ' ';

            }

            std::cout << '\n';

        }

        std::cout << '\n';

    }

    matrix feedforward(matrix input){

        for(int i = 0; i < weights.size(); ++i){

            input = matrix_multiply(input,i);

            switch(activations[i]){

                case TANH:

                    for(int j = 0; j < input.size(); ++j){

                        for(int k = 0; k < weights[i].size(); ++k){ input[j][k] = tanh(input[j][k]); }

                    }

                    break;

                case SIGMOID:

                    for(int j = 0; j < input.size(); ++j){

                        for(int k = 0; k < weights[i].size(); ++k) input[j][k] = sigmoid(input[j][k]);

                    }

                    break;

                case RELU:

                    break;

                    for(int j = 0; j < input.size(); ++j){

                        for(int k = 0; k < weights[i].size(); ++k) input[j][k] = relu(input[j][k]);

                    }

                case LEAKY_RELU:

                    for(int j = 0; j < input.size(); ++j){

                        for(int k = 0; k < weights[i].size(); ++k) input[j][k] = leaky_relu(input[j][k]);

                    }

                    break;

            }

        }

        return input;

    }

    T get_cost(matrix input, matrix label){

        matrix output = feedforward(input);

        T cost = 0;

        for(int i = 0; i < label.size(); ++i){

            for(int j = 0; j < label[i].size(); ++j){

                T diff = output[i][j]-label[i][j];

                cost += diff*diff;

            }

        }

        return cost/label.size();

    }

    void assign_weight(const std::vector<matrix> _weights){

        weights = _weights;

        layers.reserve(_weights.size()+1);

        layers.emplace_back(_weights[0][0].size());

        for(int i = 0; i < _weights.size(); ++i){

            layers.emplace_back(_weights[i].size());

        }

    }

    void assign_bias(const matrix _bias){

        bias = _bias;

    }

    void assign_activations(const std::vector<ACTIVATION_FUNCTIONS> layer_activations){

        activations = layer_activations;

    }

    std::vector<matrix> get_weights(){

        return weights;

    }

    matrix get_bias(){

        return bias;

    }

    void backpropagate(matrix input ,matrix training_label){

        std::vector<matrix> outputs(weights.size()+1);

        std::vector<matrix> deriv(weights.size());

        outputs[0] = input;

        for(int i = 0; i < weights.size(); ++i){

            deriv[i] = matrix(input.size(),vector(weights[i].size(),1));

            outputs[i+1] = matrix_multiply(outputs[i],i);

            switch(activations[i]){

                case TANH:

                    for(int j = 0; j < input.size(); ++j){

                        for(int k = 0; k < weights[i].size(); ++k){ 

                            outputs[i+1][j][k] = tanh(outputs[i+1][j][k]); 

                            deriv[i][j][k] = tanh_deriv(outputs[i+1][j][k]);

                        }

                    }

                    break;

                case SIGMOID:

                    for(int j = 0; j < input.size(); ++j){

                        for(int k = 0; k < weights[i].size(); ++k){  

                            outputs[i+1][j][k] = sigmoid(outputs[i+1][j][k]);

                            deriv[i][j][k] = sigmoid_deriv(outputs[i+1][j][k]);

                        }

                    }

                    break;

                case RELU:

                    break;

                    for(int j = 0; j < input.size(); ++j){

                        for(int k = 0; k < weights[i].size(); ++k) {

                            deriv[i][j][k] = relu_deriv(outputs[i+1][j][k]);

                            outputs[i+1][j][k] = relu(outputs[i+1][j][k]); 

                        }

                    }

                case LEAKY_RELU:

                    for(int j = 0; j < input.size(); ++j){

                        for(int k = 0; k < weights[i].size(); ++k){

                            deriv[i][j][k] = leaky_relu_deriv(outputs[i+1][j][k]); 

                            outputs[i+1][j][k] = leaky_relu(outputs[i+1][j][k]);

                        }

                    }

                    break;

            }

        }

        std::vector<matrix> weight_deriv(weights.size());

        matrix bias_deriv(weights.size());

        for(int i = 1; i < layers.size(); ++i){

            weight_deriv[i-1] = matrix(layers[i],vector(layers[i-1],0));

            bias_deriv[i-1] = vector(layers[i],0);

        }

        matrix prev_deriv(input.size(),vector(weights[weights.size()-1][0].size(),0));

        {

            for(int i = 0; i < input.size(); ++i){

                for(int j = 0; j < weights[weights.size()-1].size(); ++j){

                    T deriv_ol = 2*(training_label[i][j]-outputs[outputs.size()-1][i][j])*deriv[weights.size()-1][i][j];

                    for(int k = 0; k < weights[weights.size()-1][0].size(); ++k){

                        weight_deriv[weights.size()-1][j][k] += outputs[outputs.size()-2][i][k]*deriv_ol;

                        prev_deriv[i][k] += weights[weights.size()-1][j][k]*deriv_ol; 

                    }

                    bias_deriv[weights.size()-1][j] += deriv_ol;

                }

            }

        }

        for(int i = weights.size()-2; i >= 0; --i){

            matrix temp(input.size(),vector(weights[i][0].size(),0));

            for(int j = 0; j < input.size(); ++j){

                for(int k = 0; k < weights[i].size(); ++k){

                    T deriv_ol = prev_deriv[j][k]*deriv[i][j][k];

                    for(int h = 0; h < weights[i][0].size(); ++h){

                        weight_deriv[i][k][h] += deriv_ol * outputs[i][j][k];

                        temp[j][h] += deriv_ol*weights[i][k][h];

                    }

                    bias_deriv[i][k] += deriv_ol;

                }

            }            

            prev_deriv = temp;

        }

        for(int i = 0; i < weights.size(); ++i){

            for(int j = 0; j < weights[i].size(); ++j){

                for(int k = 0; k < weights[i][0].size(); ++k){

                    weights[i][j][k] += weight_deriv[i][j][k]/input.size() * lr;

                }

                bias[i][j] += bias_deriv[i][j]/input.size() * lr;

            }

        }

    }

private:

    T sigmoid(T n){ return 1.0/(1.0+std::exp(-n)); }

    T tanh(T n) { return std::tanh(n); }

    T relu(T n) { return n>0?n:0; }

    T leaky_relu(T n) { return n>0?n:n*0.01; }

    T sigmoid_deriv(T n) {return n*(1.0-n); }

    T tanh_deriv(T n) { return 1.0-n*n; }

    T relu_deriv(T n) { return n>0?1:0; }

    T leaky_relu_deriv(T n) { return n>0?1:0.01; }

    matrix matrix_multiply(const matrix &in, int index){

        matrix output(in.size(),vector(layers[index+1],0));

        for(int i = 0; i < in.size(); ++i){

            for(int j = 0; j < layers[index+1]; ++j){

                output[i][j] = inner_product(weights[index][j].begin(),weights[index][j].end(),in[i].begin(),bias[index][j]);

            }

        }

        return output;

    }

    matrix bias;

    std::vector<matrix> weights;

    std::vector<int> layers;

    std::vector<ACTIVATION_FUNCTIONS> activations;

    T lr = 0.1;

};

neural_network_vector<double> net(2,{2,1},{TANH,TANH});

using namespace std;

int window_width = 500, window_height = 500;

int rows = 20, cols = 20;

int space_x = window_width / rows, space_y = window_height / cols;

SDL_Window *window;

SDL_Renderer *window_renderer;

vector<vector<double>> input;

vector<vector<double>> training_input = {

    {-1,-1},

    { 1,-1},

    {-1, 1},

    { 1, 1}

};

vector<vector<double>> training_label = {

    {-1},

    { 1},

    { 1},

    {-1}

};

bool initialize(){

    input.reserve(window_width*window_height);

    for(int i = 0; i < cols; ++i){

        for(int j = 0; j < rows; ++j){

            input.push_back({(i/(double)cols)*2.0-1.0,(j/(double)rows)*2.0-1.0});

        }

    }

    if(SDL_Init(SDL_INIT_VIDEO) < 0){

        cerr << "Error Initializing SDL " << SDL_GetError();

        return false;

    } 

    window = SDL_CreateWindow("Test Window",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,window_width,window_height,SDL_WINDOW_SHOWN);

    if(!window){

        cerr << "Error Creating Window\n";

        return false;

    }

    window_renderer = SDL_CreateRenderer(window,-1,SDL_RENDERER_ACCELERATED);

    if(!window_renderer){

        cerr << "Error creating renderer\n";

        return false;

    }

    

    SDL_DisplayMode display;

    SDL_GetCurrentDisplayMode(0,&display);

    window_width = display.w;

    window_height = display.h;

    

    space_x = window_width / rows;   

    space_y = window_height / cols;

    

    return true;

}

void main_loop(){

    bool quit = 0;

    SDL_Event event;

    while(!quit){

        while(SDL_PollEvent(&event) != 0){

            if(event.type == SDL_QUIT){

                quit = 1;

                break;

            }else if(event.type == SDL_KEYDOWN){

                switch(event.key.keysym.sym){

                    case SDLK_ESCAPE:

                        quit = 1;

                        break;

                }

            }

        }

        SDL_SetRenderDrawColor(window_renderer,0xFF,0xFF,0xFF,0xFF);

        SDL_RenderClear(window_renderer);

        net.backpropagate(training_input,training_label);

        vector<vector<double>> output = net.feedforward(input);

        for(int i = 0; i < rows; ++i){

            for(int j = 0; j < cols; ++j){

                SDL_Rect rect = {.x=j*space_x,.y=i*space_y,.w=space_x,.h=space_y};

                double c = output[i*cols+j][0]*0.5+0.5;

                SDL_SetRenderDrawColor(window_renderer,c*255,c*255,c*255,255);

                SDL_RenderFillRect(window_renderer,&rect);

            }

        }

        SDL_RenderPresent(window_renderer);

    }

}

void close(){

    SDL_DestroyRenderer(window_renderer);

    window_renderer = NULL;

    SDL_DestroyWindow(window);

    window = NULL;

    SDL_Quit();

}

int main(){

	initialize();	main_loop();

	close();

	return 0;

}
