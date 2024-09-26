// algo.h
#ifndef ALGO_H
#define ALGO_H

#include <vector>
#include <iostream>
#include <limits>

using namespace std;

class SegmentTree {
private:
    vector<int> tree;
    int size;

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node + 1, start, mid);
            build(arr, 2 * node + 2, mid + 1, end);
            tree[node] = min(tree[2 * node + 1], tree[2 * node + 2]);
        }
    }

public:
    SegmentTree(const vector<int>& arr) {
        size = arr.size();
        tree.resize(4 * size);
        build(arr, 0, 0, size - 1);
    }

    int query(int l, int r, int node, int start, int end) {
        if (r < start || end < l) return INT_MAX;
        if (l <= start && end <= r) return tree[node];
        int mid = (start + end) / 2;
        int left_query = query(l, r, 2 * node + 1, start, mid);
        int right_query = query(l, r, 2 * node + 2, mid + 1, end);
        return min(left_query, right_query);
    }

    int query(int l, int r) {
        return query(l, r, 0, 0, size - 1);
    }
};

#define SGT SegmentTree

#endif // ALGO_H

