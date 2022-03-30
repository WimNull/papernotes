# papernotes
这是一个论文笔记仓库


```c++
int KMP(const char* t, const char* p)
{
    int* next = (int*)malloc(strlen(p) * sizeof(int));
    int times = 0;
    getNext(p, next);
    int i = 0, j = 0, tlen = strlen(t), plen = strlen(p);
    while (i < tlen) {
        if (j == -1 || t[i] == p[j]) {
            i++;
            j++;
        } else
            j = next[j]; 
        if (j == plen) {
            times++;
            j = next[j];
        }
    }
    return times;
}
```