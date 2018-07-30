/* 
   CNN-LSTM-CTC-OCR
   Copyright (C) 2018 Benjamin Gafford, Ziwen Chen

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <boost/lockfree/queue.hpp>
#include <thread>
#include <string>
#include <vector>
#include <fstream>
#include <mtsynth/map_text_synthesizer.hpp>
#include <mtsynth/mts_utilities.hpp>
#include <stdio.h>
#include <pthread.h>
#include <signal.h>

// Num elements to hold in g_data_pool queue
#define BUFFER_SIZE 100
#define NUM_PRODUCERS 4

// producer threads keep producing while nonzero
int g_keep_producing = 1;

// Contains all of the raw data of a sample
typedef struct sample {
  unsigned char* img_data;
  size_t height;
  size_t width;
  char* caption;
} sample_t;

// For ctypes visibility
extern "C" {
  unsigned char* get_img_data(void* ptr);
  size_t get_height(void* ptr);
  size_t get_width(void* ptr);
  char* get_caption(void* ptr);
  void mts_init(void);
  void* get_sample(void);
  void free_sample(void* ptr);
  void mts_cleanup(void);
}

void free_sample(void* ptr) {
  sample_t* s = (sample_t*)ptr;
  free(s->img_data);
  free(s->caption);
  free(s);
}

unsigned char* get_img_data(void* ptr) {
  return ((sample_t*)ptr)->img_data;
}

size_t get_height(void* ptr) {
  return ((sample_t*)ptr)->height;
}
size_t get_width(void* ptr) {
  return ((sample_t*)ptr)->width;
}
char* get_caption(void* ptr) {
  return ((sample_t*)ptr)->caption;
}

// Queue to manage concurrent producing/consuming
boost::lockfree::queue<sample_t*> g_data_pool(BUFFER_SIZE);

// Maintain producer threads
std::vector<std::thread> g_producer_threads;

/* Get lexicon from a given file -- Will later be internal to MTS*/
void read_words(string path, vector<String> &caps){
  ifstream infile(path);
  string line;
  while(std::getline(infile, line)) {   
      caps.push_back(String(line));
    }   
}

/* Prepare synthesizer object for synthesis w/ params */
void prepare_synthesis(cv::Ptr<MapTextSynthesizer> s) {
  vector<String> caps;

  //TODO fix this
  read_words("/home/gaffordb/new_mts/MapTextSynthesizer/samples/IA/Civil.txt",caps);
  
  vector<String> blocky;
  blocky.push_back("MathJax_Fraktur");
  blocky.push_back("eufm10");

  vector<String> regular;
  regular.push_back("cmmi10");
  regular.push_back("Sans");
  regular.push_back("Serif");

  vector<String> cursive;
  cursive.push_back("URW Chancery L");

  //s->setSampleCaptions(caps);
  s->setBlockyFonts(blocky);
  s->setRegularFonts(regular);
  s->setCursiveFonts(cursive);
}

/* data synthesis function. puts data into g_data_pool */
void synthesize_data(shared_ptr<unordered_map<string, double> > params) {   
  auto synthesizer = MapTextSynthesizer::create(params);
  prepare_synthesis(synthesizer);

  String label;
  Mat image;

  while(g_keep_producing) {

    // Fill in label, image
    synthesizer->generateSample(label, image);

    // Stick the necessary data into sample_t struct
    sample_t* spl = (sample_t*)malloc(sizeof(sample_t));

    size_t num_elements = image.rows * image.cols;
    size_t buff_size = num_elements * sizeof(unsigned char); 

    //allocate enough space for img_data
    if(!(spl->img_data = (unsigned char*)malloc(buff_size))) {
      perror("failed to allocate image data!\n");
    }

    //copy into image data 
    memcpy(spl->img_data, image.data, buff_size);
    spl->height = image.rows;
    spl->width = image.cols;

    //
    if(!(spl->caption = strdup(((std::string)label).c_str()))) {
      mts_cleanup();
      perror("failed to allocate memory for caption!\n");
    }

    /* If queue is full, maybe sleep a little bit??
     * Note: cv would probably be more appropriate here 
     */
    while(!g_data_pool.push(spl)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
}
  
/* Run all of the threads for producing*/
void run_producers(void) {

  MTS_Utilities utils = MTS_Utilities();

  for(int i = 0; i < NUM_PRODUCERS; i++) {
    g_producer_threads.push_back(std::thread(synthesize_data, utils.params));
  }
}

/* Consumer function for use via python */
void* get_sample(void) {
  sample_t* data;

  //Get next sample
  while(!g_data_pool.pop(data)) {
    /*Failed to pop, wait maybe? (again, cv more appropriate. do later) */
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  return (void*)data;
}

/* For less elegant cleanup */
void mts_force_cleanup(void) {
  //tell producers to knock it off
  g_keep_producing = 0;
  
  //join all producer threads
  for(auto& thd : g_producer_threads) {
    thd.detach();
  }
}

/* SIGSEGV handler. Re-exec program */
void segfault_sigaction(int signal, siginfo_t* si, void* arg) {
  printf("Caught segfault.\n");
  mts_force_cleanup();
  
  //NOTE: these should probably not be hardcoded like this
  char* py_path = "/home/gaffordb/venv/tf-1.8/bin/python";
  char* exec_path = "/home/gaffordb/new_mts/MapTextSynthesizer/samples/mts_buff/mts_faster.py";
  
  char* argv[3];
  argv[0] = py_path;
  argv[1] = exec_path;
  argv[2] = NULL;
  
  //restart program
  execv(argv[0], argv);

  //for sanity
  printf("We should not get here.\n");
  exit(1);
}

/* Call if you want to try to recover from segfault (ie restart synth) */
void set_sigsegv_handler() {
  struct sigaction sa;
  memset(&sa, 0, sizeof(struct sigaction));
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = segfault_sigaction;
  sa.sa_flags = SA_SIGINFO;

  //create SIGSEGV handler
  sigaction(SIGSEGV, &sa, NULL);
}

/* Called before using python generator function */
void mts_init(void) {
  //set_sigsegv_handler(); -- uncomment for segfault recovery
  run_producers();
}


/* Called after using python generator function */
void mts_cleanup(void) {
  //tell producers to knock it off
  g_keep_producing = 0;
  
  //join all producer threads
  for(auto& thd : g_producer_threads) {
    thd.join();
  }
}


