#ifndef __NE16_PARAMS_H
#define __NE16_PARAMS_H

#include <ne16.h>
#include <ne16_task.h>
#include <ne16_pulp_bsp.h>
#include <ne16_monitor.h>

#define DB_BUFFER_SIZE 2
#define NE16_TASKS 3
#define LOADER_TASK (0)
#define EXECUTE_TASK (1)
#define STORER_TASK (2)
#define SINGLE_CORE_TASK (3)

typedef struct nnx_monitor_s{
    Monitor input, output;
} nnx_monitor_t;

static nnx_monitor_t nnx_monitor;

static ne16_dev_t* nnx_dev;
static ne16_task_t nnx_tasks[DB_BUFFER_SIZE];
static unsigned int nnx_db_O[NE16_TASKS]={0,0,0};
static unsigned int nnx_db_I[NE16_TASKS]={0,0,0};
static unsigned int nnx_db_W[NE16_TASKS]={0,0,0};

static unsigned int nnx_input_loaded=0;

void match_ne16_set_nnx_dev();

ne16_dev_t* match_ne16_get_nnx_dev();

ne16_task_t* match_ne16_get_nnx_task(int n);

unsigned int get_nnx_db_O(int n);

unsigned int get_nnx_db_I(int n);

unsigned int get_nnx_db_W(int n);

void inc_nnx_db_O(int n);

void inc_nnx_db_I(int n);

void inc_nnx_db_W(int n);

nnx_monitor_t* get_nnx_monitor();

#endif