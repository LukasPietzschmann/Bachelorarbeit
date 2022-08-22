#include "oper_map.h"

#include <assert.h>
#include <stdlib.h>
#ifdef RUNTIME_DEBUG_OUTPUT
#include <stdio.h>
#endif

#define MAP_SIZE 512

ENTRY entries[MAP_SIZE];
int next_entries_index = 0;

void init_map() {
	int result = hcreate(MAP_SIZE);
	assert(result != 0 && "Creating a hash-map should not fail!");
}

void destroy_map() {
	hdestroy();
	for(int i = 0; i < next_entries_index; ++i) {
		free(entries[i].key);
		free(entries[i].data);
	}
}

// Der gesuchte Key kann mit alloca alloziert werden und muss nicht mit free freigegeben werden
ENTRY* search(char* key) {
	ENTRY e;
	e.key = key;
	e.data = NULL;
	ENTRY* res = hsearch(e, FIND);
#ifdef RUNTIME_DEBUG_OUTPUT
	if(res == NULL)
		printf("\"%s\" wurde nicht gefunden\n", key);
	else
		printf("\"%s\" wurde gefunden\n", key);
#endif
	return res;
}

// key und data müssen mit malloc erstellt werden und mit free wieder freigegeben werden
void enter(char* key, void* data) {
	ENTRY e;
	e.key = key;
	e.data = data;
	entries[next_entries_index++] = e;
	hsearch(e, ENTER);
}
