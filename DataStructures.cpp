template<typename T>
class Treap {
private:
    struct TreapNode {
        int pri, size, reverse;
        T key, ans, lazy;
        ll pref[2], suff[2];
        TreapNode* left;
        TreapNode* right;
        
		TreapNode(T key) : reverse(0), key(key), ans(key), pri(rand()), size(1), left(nullptr), right(nullptr) {
            for(int i = 0; i < HASH_COUNT; i++) {
                pref[i] = suff[i] = key;
            }
        }

    };

    TreapNode* root;

    int size(TreapNode* treap) {
        return !treap ? 0 : treap->size;
    }
    
    T get_ans(TreapNode* treap) {
        return !treap ? 0 : treap->ans;
    }
    
    ll get_pref(TreapNode* treap, int i) {
        return !treap ? 0 : (treap->reverse ? treap->suff[i] : treap->pref[i]);
    }

    ll get_suff(TreapNode* treap, int i) {
        return !treap ? 0 : (treap->reverse ? treap->pref[i] : treap->suff[i]);
    }
    
	void unite(TreapNode* treap) {  
        if(!treap) return;
        treap->size = size(treap->left) + size(treap->right) + 1;
//        for(int i = 0; i < HASH_COUNT; i++) {
//            treap->pref[i] = (get_pref(treap->right, i) + (p[i][size(treap->right)] * treap->key) % mod[i] + (get_pref(treap->left, i) * p[i][size(treap->right) + 1]) % mod[i]) % mod[i];
//            treap->suff[i] = (get_suff(treap->left, i) + (p[i][size(treap->left)] * treap->key) % mod[i] + (get_suff(treap->right, i) * p[i][size(treap->left) + 1]) % mod[i]) % mod[i];
//        }
    }


	void apply(TreapNode*treap, int flag, T val = 0) {
        if(!treap) return;
        treap->key += val;
        treap->lazy += val;
        treap->reverse ^= flag;
    }

    void push(TreapNode* treap) {  
        if(!treap) return;
        if(treap->reverse) swap(treap->left, treap->right);
        apply(treap->left, treap->reverse, treap->lazy);
        apply(treap->right, treap->reverse, treap->lazy);
        treap->lazy = 0;
        treap->reverse = 0;
        unite(treap);
    }

    void split(TreapNode* treap, TreapNode*& left, TreapNode*& right, int k) {
        if (!treap) {
            left = right = nullptr;
            return;
        }
        push(treap);
        if ((by_value ? (treap->key > k) : (size(treap->left) >= k))) { 
            split(treap->left, left, treap->left, k);
            right = treap;
        } else {
            split(treap->right, treap->right, right, k - (by_value ? 0 : (size(treap->left) + 1)));
            left = treap;
        }
        unite(treap);
    }

	void merge(TreapNode*& treap, TreapNode* left, TreapNode* right) {
        if (!left || !right) {
            treap = left ? left : right;
            return;
        }
        if(left->pri < right->pri) {
            push(left);
            merge(left->right, left->right, right);
            treap = left;
        } else {
            push(right);
            merge(right->left, left, right->left);
            treap = right;
        }
        unite(treap);
    }

    void destroy(TreapNode* treap) {
        if (!treap) return;
        destroy(treap->left);
        destroy(treap->right);
        delete treap;
    }

public:
    bool by_value;
    Treap(bool by_value = false) : root(nullptr), by_value(by_value) {} // all one base indexing
    
    ~Treap() {
        destroy(root);
    }

    void insert(T key) { 
        if(!by_value) {
            merge(root, root, new TreapNode(key));
            return;
        }
        TreapNode* A;
        split(root, root, A, key - 1);
        merge(root, root, new TreapNode(key));
        merge(root, root, A);
    }
    
	void insert_at(int k, T x) {
        if(size(root) < k) {
            insert(x);
            return;
        }
        TreapNode* A;
        split(root, root, A, k - 1);
        merge(root, root, new TreapNode(x));
        merge(root, root, A);
    }

    void split_and_insert_at(int l, int r, int k) {
        // split s[l, r], concatnate t = s[1, l - 1] + s[r + 1, n], then insert s[i, j] at kth position of the t string
        TreapNode *A, *B;
        split(root, root, A, l - 1);
        split(A, A, B, r - l + 1);
        merge(root, root, B);
        if(size(root) < k) {
            merge(root, root, A);
            return;
        }
        split(root, root, B, k);
        merge(root, root, A);
        merge(root, root, B);
    }

	bool is_palindrome(int l, int r) { // https://csacademy.com/contest/archive/task/strings/statement/
        TreapNode *L = nullptr, *M = nullptr, *R = nullptr;
        split(root, L, M, l - 1);
        split(M, M, R, r - l + 1);
        push(M);  
        bool ok = (M->pref[0] == M->suff[0]) && (M->pref[1] == M->suff[1]);
        merge(M, M, R);
        merge(root, L, M);
        return ok;
    }

	void split_and_apply(int l, int r, T k = 0) { 
        if(by_value) {
            TreapNode* A, *B;
            split(root, root, A, l - 1); 
            if(A) { 
                apply(A, 1, k);
                push(A);
            }
            split(A, A, B, l - 1);
            root = merge_treap(root, A);
            merge(root, root, B);
            return;
        }
        TreapNode* A, *B;
        split(root, root, A, l - 1); 
        split(A, A, B, r - l + 1);
        if(A) { 
            apply(A, 1, k);
            push(A);
        }
        merge(root, root, A);
        merge(root, root, B);
    }


	void split_and_swap(int k) {
        if(k == 0 || k == size(root)) return; 
        TreapNode* A, *B, *C;
        split(root, root, A, k);
        if(!A) return;
        merge(root, A, root);
    }

    void shift_right(int l, int r) { // [1 2 3 4] -> [4 1 2 3] and off_set by 1
        r = r - l + 1;
        TreapNode* A, *B, *C;
        split(root, root, A, l - 1);
        split(A, A, B, r);
        split(A, A, C, r - 1);
        merge(root, root, C);
        merge(root, root, A);
        merge(root, root, B);
    }

    void erase_at(int k) { 
        assert(size(root) >= k);
        TreapNode*A, *B;
        split(root, root, A, k - 1);
        split(A, A, B, 1);
        merge(root, root, B);
    }
	
	void update_at(int k, T x) {
        if(size(root) < k) return;
        TreapNode*A, *B;
        split(root, root, A, k - 1);
        split(A, A, B, 1);
        A = new TreapNode(x);
        merge(root, root, A);
        merge(root, root, B);
    }

    T queries_at(int k) {  
        TreapNode* A, *B, *C;   
        split(root, A, B, k - 1);
        split(B, B, C, 1);  
        T ans = B->key;
        merge(root, A, B);  
        merge(root, root, C);
        return ans;
    }
	
	T queries_range(int l, int r) {
        TreapNode*A, *B;
        split(root, root, A, l - 1);
        split(A, A, B, (!by_value ? r - l + 1 : r));
        T res = get_ans(A);
        merge(root, root, A);
        merge(root, root, B);
        return res == INF ? -1 : res;
    }

	TreapNode* erase_range(int l, int r) {
        TreapNode* A, *B;
        split(root, root, A, l - 1);
        split(A, A, B, r);
        merge(root, root, B);
        return A;
    }

	TreapNode* merge_treap(TreapNode* A, TreapNode* B) {
        if (!B) return A;
        if (!A) return B;
        push(B);
        A = merge_treap(A, B->left);
        A = merge_treap(A, B->right);
        B->left = B->right = nullptr;
        TreapNode *L = nullptr, *R = nullptr;
        split(A, L, R, B->key);
        merge(L, L, B);
        merge(A, L, R);
        unite(A);
        return A;
    }


    void merge_treap(TreapNode* other) {
        root = merge_treap(root, other);
    }
    
    void print() {  
        print(root);
        cout << endl;
    }
	
    void print(TreapNode* treap) {  
        push(treap);
        if(!treap) return;
        print(treap->left); 
        cout << char(treap->key + 'a');
        print(treap->right);
    }
	
	void print_substring(int pos, int len) { // 1 base index
        TreapNode*A, *B;
        split(root, root, A, pos - 1);
        split(A, A, B, len);
        print(A);
        cout << endl;
        merge(root, root, A);
        merge(root, root, B);
    }
};

int L[MX], R[MX], ans[MX], lazy1[MX], lazy2[MX], id[MX], ptr, key[MX], res[MX];
ull pri[MX];
int new_node(int K, int Id) {
    int node = ++ptr;
    L[ptr] = R[ptr] = ans[ptr] = lazy1[ptr] = lazy2[ptr] = 0;
    pri[ptr] = rng();
    key[ptr] = K;
    id[ptr] = Id;
    return node;
}
template<typename T>
class Treap {
private:
    int root = 0;
 
	inline void apply(int treap, int c, T val = 0) {
        if(!treap) return;
        key[treap] += val;
        ans[treap] += c;
        lazy2[treap] += c;
        lazy1[treap] += val;
    }
 
    inline void push(int treap) {  
        if(!treap || lazy2[treap] == 0) return;
        apply(L[treap], lazy2[treap], lazy1[treap]);
        apply(R[treap], lazy2[treap], lazy1[treap]);
        lazy2[treap] = lazy1[treap] = 0;
    }
 
    inline void split(int treap, int& left, int& right, int k) {
        if (!treap) {
            left = right = 0;
            return;
        }
        push(treap);
        if ((key[treap] > k)) { 
            split(L[treap], left, L[treap], k);
            right = treap;
        } else {
            split(R[treap], R[treap], right, k);
            left = treap;
        }
    }
 
	inline void merge(int& treap, int left, int right) {
        if (!left || !right) {
            treap = left ? left : right;
            return;
        }
        if(pri[left] < pri[right]) {
            push(left);
            merge(R[left], R[left], right);
            treap = left;
        } else {
            push(right);
            merge(L[right], left, L[right]);
            treap = right;
        }
    }
 
public:
    bool by_value;
    Treap(bool by_value = false) : root(0), by_value(by_value) {} // all one base indexing
 
    inline void insert(T key, int id) { 
        int A = 0;
        split(root, root, A, key - 1);
        merge(root, root, new_node(key, id));
        merge(root, root, A);
    }
 
	inline void split_and_apply(int l, T k = 0) { 
        int A = 0, B = 0;
        split(root, root, A, l - 1); 
        if(A) { 
            apply(A, 1, k);
            push(A);
        }
        split(A, A, B, l - 1);
        root = merge_treap(root, A);
        merge(root, root, B);
    }
 
 
    int merge_treap(int A, int B) {
        if (!B) return A;
        push(B);
        A = merge_treap(A, L[B]);
        A = merge_treap(A, R[B]);
        L[B] = R[B] = 0;
        int tx = 0, ty = 0;
        split(A, tx, ty, key[B]);
        merge(tx, tx, B);
        merge(A, tx, ty);
        return A;
    }
    
    inline void print() {  
        print(root);
    }
	
    inline void print(int treap) {  
        push(treap);
        if(!treap) return;
        print(L[treap]); 
        res[id[treap]] = ans[treap];
        print(R[treap]);
    }
};

template<class T, typename F = function<T(const T&, const T&)>>
class FW {  
    public: 
    int n, N;
    vt<T> root;    
    T DEFAULT;
    F func;
    FW() {}
    FW(int n, T DEFAULT, F func = [](const T& a, const T& b) {return a + b;}) : func(func) { 
        this->n = n;    
        this->DEFAULT = DEFAULT;
        N = log2(n);
        root.rsz(n, DEFAULT);
    }
    
    inline void update_at(int id, T val) {  
        assert(id >= 0);
        while(id < n) {    
            root[id] = func(root[id], val);
            id |= (id + 1);
        }
    }
    
    inline T get(int id) {   
        assert(id < n);
        T res = DEFAULT;
        while(id >= 0) { 
            res = func(res, root[id]);
            id = (id & (id + 1)) - 1;
        }
        return res;
    }

    inline T queries_range(int left, int right) {  
        return get(right) - get(left - 1);
    }

    inline T queries_at(int i) {
        return queries_range(i, i);
    }

    inline void update_range(int l, int r, T val) {
        update_at(l, val), update_at(r + 1, -val);
    }
	
	inline void reset() {
		root.assign(n, DEFAULT);
	}

    int select(int x) { // get pos where sum >= x
        int global = get(n), curr = 0;
        for(int i = N; i >= 0; i--) {
            int t = curr ^ (1LL << i);
            if(t < n && global - root[t] >= x) {
                swap(curr, t);
                global -= root[curr];
            }
        }
        return curr + 1;
    }
};

template<typename T>
struct range_fenwick {
    int n; 
    FW<T> B1, B2;
    range_fenwick(int n): n(n), B1(n, 0), B2(n, 0) {}

    inline void update_range(int l, int r, T v){
        B1.update_at(l, v);        B1.update_at(r + 1, -v);
        B2.update_at(l, v * (l - 1));  B2.update_at(r + 1, -v * r);
    }

    inline T prefix(int i){
        return i * B1.get(i) - B2.get(i);
    }

    inline T queries_range(int l,int r){
        return prefix(r) - prefix(l - 1);
    }

    inline T queries_at(int p) {
        return prefix(p) - prefix(p - 1);
    }
};

template<class T>
class FW_2D {
    public:
    int n;
    vt<vt<T>> root;
	vvi coord;
    T DEFAULT;
    FW_2D(int n, T DEFAULT) : n(n), DEFAULT(DEFAULT) {
        coord.rsz(n), root.rsz(n);
    }
 
    void go_up(int& id) {
        id |= (id + 1);
    }
 
    void go_down(int& id) {
        id = (id & (id + 1)) - 1;
    }
 
    void add_coord(int i, int x, bool is_up = true) {
        while(i >= 0 && i < n) {
            coord[i].pb(x);
            if(is_up) go_up(i);
            else go_down(i);
        }
    }
 
    void update_coord(int i, int l, int r, bool is_up = true) {
        add_coord(i, l - 1, is_up);
        add_coord(i, r, is_up);
    }

    void add_rectangle(int r1, int c1, int r2, int c2, bool is_up = true) {
        add_coord(r1, c1, is_up);
        add_coord(r1, c2 + 1, is_up);
        add_coord(r2 + 1, c1, is_up);
        add_coord(r2 + 1, c2 + 1, is_up);
    }

    void add_point(int r, int c, bool is_up = false) { // for queries on a specific point so is_up is false
        update_coord(r, c, c, is_up);
    }
 
    void build() {
        for(int i = 0; i < n; i++) {
            srtU(coord[i]);
            root[i].rsz(coord[i].size(), DEFAULT);
        }
    }
 
    int get_id(int i, int x) {
        return int(lb(all(coord[i]), x) - begin(coord[i]));
    }
 
    void update_at(int i, int x, T delta) {
        while(i < n) {
            int p = get_id(i, x);
            while(p < coord[i].size()) {
                root[i][p] = merge(root[i][p], delta);
                go_up(p);
            }
            go_up(i);
        }
    }
 
    void update_range(int i, int l, int r, T v) {
        update_at(i, l, v); 
        update_at(i, r + 1, -v);
    }
 
    void update_rectangle(int r1, int c1, int r2, int c2, T v) {
        update_range(r1, c1, c2, v);
        update_range(r2 + 1, c1, c2, -v);
    }
 
    T point_query(int i, int x) {
        T res = DEFAULT;
        while(i >= 0) {
            int p = get_id(i, x);
            while(p >= 0) {
                res = merge(res, root[i][p]);
                go_down(p);
            }
            go_down(i);
        }
        return res;
    }
 
    T bit_range_queries(int i, int low, int high) {
        if(low > high) return DEFAULT;
        return point_query(i, high) - point_query(i, low - 1);
    }
 
    T range_queries(int l, int r, int low, int high) {
        if(l > r || low > high) return DEFAULT;
        return bit_range_queries(r, low, high) - bit_range_queries(l - 1, low, high);
    }

    T merge(T left, T right) {
    }
};

template<class T, typename F = function<T(const T&, const T&)>>
class compress_FW {  
    public: 
    int n;
    vt<T> root;    
    T DEFAULT;
    F func;
    vll coord;
    compress_FW() {}
    compress_FW(T DEFAULT, F func) : func(func), DEFAULT(DEFAULT) {}
    
    void build() {
        srtU(coord);
        n = coord.size();
        root.rsz(n + 1, DEFAULT);
    }

    void add_coord(ll x) {
        coord.pb(x);
    }

    int get_id(ll x) {
        return int(lb(all(coord), x) - begin(coord));
    }

    void update_at(ll id, T val) {  
        id = get_id(id);
        while(id < n) {    
            root[id] = func(root[id], val);
            id |= (id + 1);
        }
    }
    
    T get(ll id) {   
        id = min(n - 1, get_id(id + 1) - 1);
        T res = DEFAULT;
        while(id >= 0) { 
            assert(id < n);
            res = func(res, root[id]);
            id = (id & (id + 1)) - 1;
        }
        return res;
    }

    T queries_range(ll left, ll right) {  
        return get(right) - get(left - 1);
    }

    T queries_at(ll i) {
        return queries_range(i, i);
    }

    void update_range(ll l, ll r, T val) {
        update_at(l, val), update_at(r + 1, -val);
    }
	
	void reset() {
		root.assign(n, DEFAULT);
	}

    int select(int x) { // get pos where sum >= x
        int global = get(n - 1), curr = 0;
        for(int i = log2(n) - 1; i >= 0; i--) {
            int t = curr ^ (1LL << i);
            if(t < n && global - root[t] >= x) {
                swap(curr, t);
                global -= root[curr];
            }
        }
        return curr + 1;
    }
};

template<typename T> // for queries and updating rectangle
struct BIT2D {
    int n, m;
    vt<vt<T>> B1, B2, B3, B4;
    
    BIT2D(int n, int m) : n(n), m(m) {
        B1.assign(n + 1, vt<T>(m + 1, 0));
        B2.assign(n + 1, vt<T>(m + 1, 0));
        B3.assign(n + 1, vt<T>(m + 1, 0));
        B4.assign(n + 1, vt<T>(m + 1, 0));
    }
    
    void add(vt<vt<T>> &B, int x, int y, T v) {
        for (int i = x; i <= n; i += i & -i)
            for (int j = y; j <= m; j += j & -j)
                B[i][j] += v;
    }
    
    void update_range(int x1, int y1, int x2, int y2, T v) {
        add(B1, x1, y1, v);
        add(B1, x1, y2 + 1, -v);
        add(B1, x2 + 1, y1, -v);
        add(B1, x2 + 1, y2 + 1, v);
        
        add(B2, x1, y1, v * (x1 - 1));
        add(B2, x1, y2 + 1, -v * (x1 - 1));
        add(B2, x2 + 1, y1, -v * x2);
        add(B2, x2 + 1, y2 + 1, v * x2);
        
        add(B3, x1, y1, v * (y1 - 1));
        add(B3, x1, y2 + 1, -v * y2);
        add(B3, x2 + 1, y1, -v * (y1 - 1));
        add(B3, x2 + 1, y2 + 1, v * y2);
        
        add(B4, x1, y1, v * (x1 - 1) * (y1 - 1));
        add(B4, x1, y2 + 1, -v * (x1 - 1) * y2);
        add(B4, x2 + 1, y1, -v * x2 * (y1 - 1));
        add(B4, x2 + 1, y2 + 1, v * x2 * y2);
    }
    
    T query(vt<vt<T>> &B, int x, int y) {
        T sum = 0;
        for (int i = x; i > 0; i -= i & -i)
            for (int j = y; j > 0; j -= j & -j)
                sum += B[i][j];
        return sum;
    }
    
    T prefix_sum(int x, int y) {
        return query(B1, x, y) * x * y
             - query(B2, x, y) * y
             - query(B3, x, y) * x
             + query(B4, x, y);
    }
    
    T queries_range(int x1, int y1, int x2, int y2) {
        return prefix_sum(x2, y2)
             - prefix_sum(x1 - 1, y2)
             - prefix_sum(x2, y1 - 1)
             + prefix_sum(x1 - 1, y1 - 1);
    }
    
    void update_at(int r, int c, T v) {
        update_range(r, c, r, c, v);
    }
    
    T queries_at(int r, int c) {
        return queries_range(r, c, r, c);
    }
};

template<class T>
struct BIT2D_XOR {
    int n, m;
    vector<vector<T>> data0, data1, data2, data3;
    BIT2D_XOR() { }
    BIT2D_XOR(int n, int m) : n(n), m(m), data0(n, vector<T>(m, 0)), data1(n, vector<T>(m, 0)), data2(n, vector<T>(m, 0)), data3(n, vector<T>(m, 0)) { }
    BIT2D_XOR(int n, int m, T init) : BIT2D_XOR(vector<vector<T>>(n, vector<T>(m, init))) { }
    BIT2D_XOR(const vector<vector<T>> &v) : n((int)v.size()), m((int)v[0].size()), data0(n, vector<T>(m, 0)), data1(n, vector<T>(m, 0)), data2(n, vector<T>(m, 0)), data3(v) {
        for(auto i = 1; i <= n; ++i) 
            if(i + (i & -i) <= n) 
                for(auto j = 0; j < m; ++j) 
                    data3[i + (i & -i) - 1][j] ^= data3[i - 1][j];
        for(auto i = 0; i < n; ++i) 
            for(auto j = 1; j <= m; ++j) 
                if(j + (j & -j) <= m) 
                    data3[i][j + (j & -j) - 1] ^= data3[i][j - 1];
    }
    void update_range(int xl, int xr, int yl, int yr, T x) {
        xl--, yl--;
        assert(0 <= xl && xl <= xr && xr <= n);
        assert(0 <= yl && yl <= yr && yr <= m);
        if(xl == xr || yl == yr) return;
        for(auto i = xl + 1; i <= n; i += i & -i) 
            for(auto j = yl + 1; j <= m; j += j & -j) {
                data0[i - 1][j - 1] ^= x;
                data1[i - 1][j - 1] ^= (xl & 1) * x;
                data2[i - 1][j - 1] ^= (yl & 1) * x;
                data3[i - 1][j - 1] ^= (xl & yl & 1) * x;
            }
        for(auto i = xl + 1; i <= n; i += i & -i) 
            for(auto j = yr + 1; j <= m; j += j & -j) {
                data0[i - 1][j - 1] ^= x;
                data1[i - 1][j - 1] ^= (xl & 1) * x;
                data2[i - 1][j - 1] ^= (yr & 1) * x;
                data3[i - 1][j - 1] ^= (xl & yr & 1) * x;
            }
        for(auto i = xr + 1; i <= n; i += i & -i) 
            for(auto j = yl + 1; j <= m; j += j & -j) {
                data0[i - 1][j - 1] ^= x;
                data1[i - 1][j - 1] ^= (xr & 1) * x;
                data2[i - 1][j - 1] ^= (yl & 1) * x;
                data3[i - 1][j - 1] ^= (xr & yl & 1) * x;
            }
        for(auto i = xr + 1; i <= n; i += i & -i) 
            for(auto j = yr + 1; j <= m; j += j & -j) {
                data0[i - 1][j - 1] ^= x;
                data1[i - 1][j - 1] ^= (xr & 1) * x;
                data2[i - 1][j - 1] ^= (yr & 1) * x;
                data3[i - 1][j - 1] ^= (xr & yr & 1) * x;
            }
    }
    void update_at(int x, int y, T x_val) {
        update_range(x, x + 1, y, y + 1, x_val);
    }
    T pref(int xr, int yr) const {
        assert(0 <= xr && xr <= n);
        assert(0 <= yr && yr <= m);
        T sum0 = {}, sum1 = {}, sum2 = {}, sum3 = {};
        for(auto x = xr; x > 0; x -= x & -x) 
            for(auto y = yr; y > 0; y -= y & -y) {
                sum0 ^= data0[x - 1][y - 1];
                sum1 ^= data1[x - 1][y - 1];
                sum2 ^= data2[x - 1][y - 1];
                sum3 ^= data3[x - 1][y - 1];
            }
        return (xr & yr & 1) * sum0 ^ (yr & 1) * sum1 ^ (xr & 1) * sum2 ^ sum3;
    }
    T queries_range(int xl, int xr, int yl, int yr) const {
        xl--, yl--;
        assert(0 <= xl && xl <= xr && xr <= n);
        assert(0 <= yl && yl <= yr && yr <= m);
        if(xl == xr || yl == yr) return {};
        return pref(xr, yr) ^ pref(xl, yr) ^ pref(xr, yl) ^ pref(xl, yl);
    }
    T queries_at(int x, int y) const {
        x--, y--;
        return queries_range(x, x + 1, y, y + 1);
    }
    template<class output_stream>
    friend output_stream &operator<<(output_stream &out, const BIT2D_XOR<T> &solver) {
        for(auto i = 0; i < solver.n; ++i) {
            out << "\n[";
            for(auto j = 0; j < solver.m; ++j) {
                out << solver.queries_range(i, i + 1, j, j + 1);
                if(j != solver.m - 1) out << ", ";
            }
            out << "]\n";
        }
        return out;
    }
};

// this one supports pos array
template<typename T, int BITS = 30>
struct xor_basis {
	// subsequence [l, r] having subsequence_xor of x is pow(2, (r - l + 1) - rank())
    T basis[BITS];
    int r;

    xor_basis() {
        r = 0;
        for (int b = 0; b < BITS; b++)
            basis[b] = 0;
    }

    bool insert(T x) {
        if(x == 0) return false;
        for(int b = BITS - 1; b >= 0; --b) {
            if(!have_bit(x, b)) continue;
            if(!basis[b]) {
                basis[b] = x;
                r++;
                // TODO: elimination bit, remove if WA
                for(int d = 0; d < BITS; ++d) {
                    if(d != b && ((basis[d] >> b) & 1)) {
                        basis[d] ^= x;
                    }
                }
                return true;
            }
            x ^= basis[b];
        }
        return false;
    }

    bool contains(T x) const {
        for(int b = BITS - 1; b >= 0; --b) {
            if(!have_bit(x, b)) continue;
            if(!basis[b]) return false;
            x ^= basis[b];
        }
        return true;
    }

    T min_value(T x) const {
        for(int b = BITS - 1; b >= 0; --b) {
            if(basis[b] && (x ^ basis[b]) < x)
                x ^= basis[b];
        }
        return x;
    }

    T max_value(T x = 0) const {
        for (int b = BITS - 1; b >= 0; --b) {
            if (basis[b] && (x ^ basis[b]) > x)
                x ^= basis[b];
        }
        return x;
    }

    int rank() const {
        return r;
    }

    uint64_t size() const {
        return (r >= 64 ? 0ULL : (1ULL << r));
    }

    vt<T> get_compact_basis() const {
        vt<T> vec;
        for (int b = 0; b < BITS; ++b) {
            if (basis[b]) vec.pb(basis[b]);
        }
        return vec;
    }

    T get_kth_smallest(uint64_t k) const {
        auto vec = get_compact_basis();
        int m = (int)vec.size();
        if (m >= 64 || k >= (1ULL << m)) return T(0);
        T ans = 0;
        for (int i = 0; i < m; ++i) {
            if ((k >> i) & 1ULL) ans ^= vec[i];
        }
        return ans;
    }

    T get_kth_largest(uint64_t k) const {
        uint64_t total = size();
        if (total == 0 || k >= total) return T(0);
        uint64_t idx = total - 1 - k;
        return get_kth_smallest(idx);
    }

    bool insert_base_on(T x, T c) {
        for(int b = BITS - 1; b >= 0; --b) {
            if(have_bit(c, b)) continue;
            if(!have_bit(x, b)) continue;
            if(!basis[b]) {
                basis[b] = x;
                r++;
                for(int c = 0; c < BITS; ++c)
                    if(c != b && (basis[c] >> b & 1))
                        basis[c] ^= x;
                return true;
            }
            x ^= basis[b];
        }
        return false;
    }

    T max_value_base_on(T x) const { // find max query(x) + (x ^ query(x));
        T res = 0;
        for(int b = BITS - 1; b >= 0; --b) {
            if(have_bit(x, b)) continue;
            if(!have_bit(res, b)) res ^= basis[b];
        }
        return res;
    }

	inline xor_basis operator+(const xor_basis& other) const {
        if (r == 0) return other;
        if (other.r == 0) return *this;
        const xor_basis* big   = (r >= other.r ? this  : &other);
        const xor_basis* small = (r >= other.r ? &other :  this);
        xor_basis res = *big;
        for (int i = 0; i < small->r; ++i) res.insert(small->basis[i]);
        return res;
    }


    inline xor_basis& operator^=(T k) { // https://codeforces.com/contest/587/problem/E
        for (int i = BITS - 1; i >= 0; i--)
            if (basis[i] & (1LL << 31)) { // original array
                basis[i] ^= k;
                break;
            }
        return *this;
    }

    bool operator==(const xor_basis &o) const {
        for(int b = 0; b < BITS; b++)
            if(basis[b] != o.basis[b])
                return false;
        return true;
    }
};

// this one is faster since it iterating over the # of value instead of BITS, careful with initializing BITS, you can go 60 if needed
template<typename T, int BITS = 30>
struct xor_basis {
    // subsequence [l, r] having subsequence_xor of x is pow(2, (r - l + 1) - rank())
    T basis[BITS];
    int r = 0;
 
    bool insert(T x) {
        if (x == 0) return false;
        x = min_value(x);
        if (x == 0) return false;
        for (int i = 0; i < r; ++i)
            if ((basis[i] ^ x) < basis[i]) basis[i] ^= x;
        basis[r++] = x;
        int k = r - 1;
        while (k > 0 && max_bit(basis[k]) > max_bit(basis[k - 1])) {
            swap(basis[k], basis[k - 1]);
            --k;
        }
        return true;
    }
 
    bool contains(T x) const {
        return min_value(x) == 0;
    }
 
    T min_value(T x) const {
        if(r == BITS) return 0;
        for (int i = 0; i < r; ++i) x = min(x, x ^ basis[i]);
        return x;
    }
 
    T max_value(T x = 0) const {
        for (int i = 0; i < r; ++i) x = max(x, x ^ basis[i]);
        return x;
    }
 
    int rank() const { return r; }
 
    uint64_t size() const { return (r >= 64 ? 0ULL : 1ULL << r); }
 
    vt<T> get_compact_basis() const {
        vt<T> vec;
        for (int i = 0; i < r; ++i) vec.pb(basis[i]);
        return vec;
    }
 
    T get_kth_smallest(uint64_t k) const {
        if (r >= 64 || k >= (1ULL << r)) return T(0);
        T ans = 0;
        for (int i = 0; i < r; ++i)
            if ((k >> i) & 1ULL) ans ^= basis[i];
        return ans;
    }
 
    T get_kth_largest(uint64_t k) const {
        uint64_t total = size();
        if (total == 0 || k >= total) return T(0);
        return get_kth_smallest(total - 1 - k);
    }
 
    bool insert_base_on(T x, T c) {
        for (int i = 0; i < r; ++i) x = std::min(x, x ^ basis[i]);
        if (x == 0) return false;
        for (int b = BITS - 1; b >= 0; --b) {
            if (have_bit(c, b) || !have_bit(x, b)) continue;
            for (int i = 0; i < r; ++i)
                if (have_bit(basis[i], b)) basis[i] ^= x;
            basis[r++] = x;
            int k = r - 1;
            while (k > 0 && max_bit(basis[k]) > max_bit(basis[k - 1])) {
                std::swap(basis[k], basis[k - 1]);
                --k;
            }
            return true;
        }
        return false;
    }
 
    T max_value_base_on(T x) const {
        T res = 0;
        for (int i = 0; i < r; ++i)
            if (!have_bit(x, max_bit(basis[i]))) res ^= basis[i];
        return res;
    }
 
    inline xor_basis operator+(const xor_basis& other) const {
        if (r == 0) return other;
        if (other.r == 0) return *this;
        const xor_basis* big   = (r >= other.r ? this  : &other);
        const xor_basis* small = (r >= other.r ? &other :  this);
        xor_basis res = *big;
        for (int i = 0; i < small->r; ++i) res.insert(small->basis[i]);
        return res;
    }
 
    xor_basis& operator^=(T k) {
        for (int i = r - 1; i >= 0; --i)
            if (basis[i] & 1) {
                basis[i] ^= k;
            }
        return *this;
    }
 
    bool operator==(const xor_basis &o) const {
        if (r != o.r) return false;
        for (int i = 0; i < r; ++i)
            if (basis[i] != o.basis[i]) return false;
        return true;
    }
};

class MO {  
    public: 
    int n, q;  
    int block;
    vi a;   
    var(3) Q;
    MO(vi& a, var(3)& Q) {  // 1 base index array
        n = a.size();
        q = Q.size();
        this->a = a;    
        this->Q = Q;
        block = sqrt(n);
    }

    vll queries() {    
        auto cmp = [&](const ar(3)& a, const ar(3)& b) -> bool {    
            if(a[0] / block != b[0] / block) return a[0] / block < b[0] / block;
            int d = a[0] / block;   
            if(d & 1) return a[1] > b[1];
            return a[1] < b[1];
        };
        sort(all(Q), cmp);
        vi pos(a);  
        srtU(pos); 
        const int N = pos.size();
        auto get_id = [&](ll x) -> int {
            return int(lb(all(pos), x) - begin(pos));
        };
        for(auto& x : a) x = get_id(x);
        vll dp(N);
        ll ans = 0;
        auto modify = [&](int x, int v) -> void {    
            auto& curr = dp[x];
            if(v == 1) {
                ans += curr * (curr - 1) / 2;
                curr++;
            }
            else {
                curr--;
                ans -= curr * (curr - 1) / 2;
            }
        };

        vll res(q);
        int l = 0, r = -1;    // modify to 0 as needed "left = 0"
        for(auto& [ql, qr, id] : Q) { // 1 base index
			while (r < qr) modify(a[++r], 1);
			while (l > ql) modify(a[--l], 1);
			while (r > qr) modify(a[r--], -1);
			while (l < ql) modify(a[l++], -1);
            res[id] = ans;
        }
        return res;
    }
};

template<typename T, typename F = function<T(const T&, const T&)>>
class SparseTable {
public:
    int n, m;
    vt<vt<T>> st;
    vi log_table;
    F func;
    
    SparseTable() {}

    SparseTable(const vt<T>& a, F func) : n(a.size()), func(func) {
        m = floor(log2(n)) + 1;
        st.resize(m);
        for (int j = 0; j < m; j++) st[j].resize(n - (1 << j) + 1);
        log_table.resize(n + 1);
        for (int i = 2; i <= n; i++) log_table[i] = log_table[i / 2] + 1;
        for (int i = 0; i < n; i++) st[0][i] = a[i];
        for (int j = 1; j < m; j++) {
            for (int i = 0; i + (1 << j) <= n; i++)
                st[j][i] = func(st[j - 1][i], st[j - 1][i + (1 << (j - 1))]);
        }
    }
    
    T query(int L, int R) {
        int j = log_table[R - L + 1];
        return func(st[j][L], st[j][R - (1 << j) + 1]);
    }
};

template<typename T, typename F = function<T(const T&, const T&)>>
class SparseTable2D {
public:
    int n, m, LOGN, LOGM;
    vt<vt<vt<vt<T>>>> st;
    vi logn, logm;
    F f;
    T DEFAULT;
    SparseTable2D(const vt<vt<T>> &a, T DEFAULT, F func) : f(func), DEFAULT(DEFAULT) {
        n = a.size();
        m = a[0].size();
        LOGN = floor(log2(n)) + 1;
        LOGM = floor(log2(m)) + 1;
        logn.rsz(n + 1, 0);
        logm.rsz(m + 1, 0);
        for (int i = 2; i <= n; i++) logn[i] = logn[i / 2] + 1;
        for (int j = 2; j <= m; j++) logm[j] = logm[j / 2] + 1;
        st.assign(LOGN, vt<vt<vt<T>>>(LOGM));
        for (int k = 0; k < LOGN; k++) {
            int rows = n - (1 << k) + 1;
            for (int l = 0; l < LOGM; l++) {
                int cols = m - (1 << l) + 1;
                st[k][l].rsz(rows, vt<T>(cols, DEFAULT));
            }
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                st[0][0][i][j] = a[i][j];
        for (int l = 1; l < LOGM; l++) {
            for (int i = 0; i < n; i++) {
                int cols = m - (1 << l) + 1;
                for (int j = 0; j < cols; j++)
                    st[0][l][i][j] = f(st[0][l-1][i][j], st[0][l-1][i][j + (1 << (l-1))]);
            }
        }
        for (int k = 1; k < LOGN; k++) {
            int rows = n - (1 << k) + 1;
            for (int l = 0; l < LOGM; l++) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < m - (1 << l) + 1; j++)
                        st[k][l][i][j] = f(st[k-1][l][i][j], st[k-1][l][i + (1 << (k-1))][j]);
                }
            }
        }
    }
    T query(int r1, int c1, int r2, int c2) {
        int h = r2 - r1 + 1, w = c2 - c1 + 1;
        int k = logn[h], l = logm[w];
        T a1 = st[k][l][r1][c1],
          a2 = st[k][l][r2 - (1 << k) + 1][c1],
          a3 = st[k][l][r1][c2 - (1 << l) + 1],
          a4 = st[k][l][r2 - (1 << k) + 1][c2 - (1 << l) + 1];
        return f(f(a1, a2), f(a3, a4));
    }
};

template <typename T, typename F = function<bool(const T&, const T&)>> // only handle max, min
struct linear_rmq {
    const vt<T>& values;
    F compare;
    vi head;
    vt<array<unsigned,2>> masks;

    linear_rmq(const vt<T>& arr, F cmp = F{})
      : values(arr), compare(cmp),
        head(arr.size()+1),
        masks(arr.size())
    {
        vi monoStack{-1};
        int n = arr.size();
        for (int i = 0; i <= n; i++) {
            int last = -1;
            while (monoStack.back() != -1 &&
                   (i == n || !compare(values[monoStack.back()], values[i])))
            {
                if (last != -1) head[last] = monoStack.back();
                unsigned diffBit = __bit_floor(unsigned(monoStack.end()[-2] + 1) ^ i);
                masks[monoStack.back()][0] = last = (i & -diffBit);
                monoStack.pop_back();
                masks[monoStack.back() + 1][1] |= diffBit;
            }
            if (last != -1) head[last] = i;
            monoStack.pb(i);
        }
        for (int i = 1; i < n; i++) {
            masks[i][1] = (masks[i][1] | masks[i-1][1])
                        & -(masks[i][0] & -masks[i][0]);
        }
    }

    T query(int L, int R) const {
        unsigned common = masks[L][1] & masks[R][1]
                        & -__bit_floor((masks[L][0] ^ masks[R][0]) | 1);
        unsigned k = masks[L][1] ^ common;
        if (k) {
            k = __bit_floor(k);
            L = head[(masks[L][0] & -k) | k];
        }
        k = masks[R][1] ^ common;
        if (k) {
            k = __bit_floor(k);
            R = head[(masks[R][0] & -k) | k];
        }
        return compare(values[L], values[R]) ? values[L] : values[R];
    }
};

template<typename T>
class TWO_DIMENSIONAL_RANGE_QUERY {   
    public: 
    vt<vt<T>> prefix;
    int n, m;
    TWO_DIMENSIONAL_RANGE_QUERY(const vvi& grid) {  
        n = grid.size(), m = grid[0].size();
        prefix.assign(n + 1, vt<T>(m + 1));  
        for(int i = 1; i <= n; i++) {  
            T sm = 0;
            for(int j = 1; j <= m; j++) {  
                sm += grid[i - 1][j - 1];
                prefix[i][j] = sm + prefix[i - 1][j];
            }
        }
    }
    
    T get(int r1, int c1, int r2, int c2) {   
        if(r2 < r1 || c2 < c1 || r1 <= 0 || r2 > n || c1 <= 0 || c2 > m) return -inf;
        T bottomRight = prefix[r2][c2];   
        T topLeft = prefix[r1 - 1][c1 - 1];
        T topRight = prefix[r1 - 1][c2];  
        T bottomLeft = prefix[r2][c1 - 1];
        return bottomRight - topRight - bottomLeft + topLeft;
    }
};

template<class T>
class Splay_Tree {
public:
    struct Node {
        Node *left, *right, *parent;
        T key;   
        int delta;  
        int need_rev;
        int offset;
        int size; 
        ll Sum[5];

        Node(int x = 0) 
            : left(nullptr), right(nullptr), parent(nullptr), key(x),
              delta(0), offset(0), size(1), need_rev(0)
        {
            mset(Sum, 0);
            Sum[0] = x;
        }

        void fix() {
            if (left)  left->parent = this;
            if (right) right->parent = this;
        }

        void push_down() {
            if(need_rev) {
                swap(left, right);
                if(left) left->need_rev ^= 1;
                if(right) right->need_rev ^= 1;
                need_rev = 0;
            }
            if(delta) {
                if (left) left->delta += delta;
                if (right) right->delta += delta;
                delta = 0;
            }
        }

        void unite() {
            memset(Sum, 0, sizeof(Sum));
            size = 1;
            if (left) {
                size += left->size;
                for (int i = 0; i < 5; i++){
                    Sum[i] = left->Sum[i];
                }
            }
            Sum[(size - 1) % 5] += key;
            if (right) {
                for (int i = 0; i < 5; i++){
                    Sum[(i + size) % 5] += right->Sum[i];
                }
                size += right->size;
            }
        }
    };

private:
    Node* root;

    void rotate_right(Node* x) {
        Node* p = x->parent;
        if (!p) return;
        p->push_down();
        x->push_down();
        Node* b = x->right;
        x->right = p;
        p->left = b;
        if (b) b->parent = p;
        x->parent = p->parent;
        if (p->parent) {
            if (p->parent->left == p) p->parent->left = x;
            else p->parent->right = x;
        }
        p->parent = x;
        p->fix();
        x->fix();
        p->unite();
        x->unite();
    }

    void rotate_left(Node* x) {
        Node* p = x->parent;
        if (!p) return;
        p->push_down();
        x->push_down();
        Node* b = x->left;
        x->left = p;
        p->right = b;
        if (b) b->parent = p;
        x->parent = p->parent;
        if (p->parent) {
            if (p->parent->left == p) p->parent->left = x;
            else p->parent->right = x;
        }
        p->parent = x;
        p->fix();
        x->fix();
        p->unite();
        x->unite();
    }

    void splay(Node* x, Node* goal = nullptr) {
        if (!x) return;
        while (x->parent != goal) {
            Node* p = x->parent;
            Node* g = p->parent;
            if (g == goal) {  // Zig step.
                if (p->left == x) rotate_right(x);
                else rotate_left(x);
            }
            else if (g->left == p && p->left == x) { // Zig-Zig
                rotate_right(p);
                rotate_right(x);
            }
            else if (g->right == p && p->right == x) { // Zig-Zig
                rotate_left(p);
                rotate_left(x);
            }
            else if (g->left == p && p->right == x) { // Zig-Zag
                rotate_left(x);
                rotate_right(x);
            }
            else { // g->right == p && p->left == x, Zig-Zag.
                rotate_right(x);
                rotate_left(x);
            }
        }
        if (goal == nullptr) root = x;
    }

public:
    Splay_Tree() : root(nullptr) {}

    void Insert(int x) {
        root = merge(root, new Node(x));
    }

    Node* merge(Node* L, Node* R) {
        if(!L) return R;
        if(!R) return L;
        Node* maxL = L;
        while (maxL->right) maxL = maxL->right; // find the right_most node of left_children
        splay(maxL);
        maxL->right = R;
        if (R) R->parent = maxL;
        maxL->unite();
        return maxL;
    }
    // Find the node with key x. Splay it to the root if found.
    Node* Find(int x) {
        Node* cur = root;
        while (cur) {
            if (x == cur->key) break;
            else if (x < cur->key) cur = cur->left;
            else cur = cur->right;
        }
        if (cur) splay(cur);
        return cur;
    }

    bool Erase(int x) {
        Node* target = Find(x);
        if (!target) return false;
        splay(target);
        Node* L = target->left;
        Node* R = target->right;
        if (L) L->parent = nullptr;
        if (R) R->parent = nullptr;
        delete target;
        root = merge(L, R);
        return true;
    }

    ll get() {
        if (!root) return 0;
        root->unite();
        return root->Sum[2];
    }

    int get_k(int k) {
        if (!root || k < 0 || k >= root->size)
            throw out_of_range("k is out of bounds");
        Node* node = get_k(root, k);
        splay(node);
        return node->key;
    }

    Node* get_k(Node* curr, int k) { // index start at 0
        int left_size = (curr->left ? curr->left->size : 0);
        if (k < left_size) return get_k(curr->left, k);
        else if (k == left_size) return curr;
        else return get_k(curr->right, k - left_size - 1);
    }
};

struct LCT {
    struct Node {
        int p = 0;
        int sz = 1;
        int c[2] = {0, 0};
        ll val = 0, sm = 0, mn = INF, mx = -INF, lazy_set = INF, lazy_add = 0;
        bool flip = false;
        Node() {}
        Node(ll v)
            : p(0), sz(1), val(v), sm(v), mn(v), mx(v),
            lazy_set(INF), lazy_add(0), flip(false) {
                c[0] = c[1] = 0;
            }
    };
    vt<Node> T;
    LCT() {}
    LCT(int N) : T(N + 1) {}
    LCT(int N, const vll& A)
      : T(N + 1) {
        for (int i = 1; i <= N; ++i) {
            T[i] = Node(A[i]);
        }
    }

    bool notRoot(int x) {
        int p = T[x].p;
        return p && (T[p].c[0] == x || T[p].c[1] == x);
    }

    void push(int x) {
        if (!x) return;
        int l = T[x].c[0], r = T[x].c[1];

        if (T[x].flip) {
            swap(T[x].c[0], T[x].c[1]);
            if (l) apply_flip(l);
            if (r) apply_flip(r);
            T[x].flip = false;
        }

        if (T[x].lazy_set != INF) {
            if (l) apply_set(l, T[x].lazy_set);
            if (r) apply_set(r, T[x].lazy_set);
            T[x].lazy_set = INF;
        }

        if (T[x].lazy_add) {
            if (l) apply_add(l, T[x].lazy_add);
            if (r) apply_add(r, T[x].lazy_add);
            T[x].lazy_add = 0;
        }
    }

    void pull(int x) {
        push(T[x].c[0]);
        push(T[x].c[1]);
        int l = T[x].c[0], r = T[x].c[1];
        T[x].sz = 1 + (l ? T[l].sz : 0) + (r ? T[r].sz : 0);
        T[x].sm = T[x].val + (l ? T[l].sm : 0) + (r ? T[r].sm : 0);
        T[x].mn = min({T[x].val, l ? T[l].mn : INF, r ? T[r].mn : INF});
        T[x].mx = max({T[x].val, l ? T[l].mx : -INF, r ? T[r].mx : -INF});
    }

    void apply_add(int x, ll v) {
        if (!x) return;
        T[x].lazy_add += v;
        T[x].val += v;
        T[x].sm += v * T[x].sz;
        T[x].mn += v;
        T[x].mx += v;
    }

    void apply_set(int x, ll v) {
        if (!x) return;
        T[x].lazy_set = v;
        T[x].lazy_add = 0;
        T[x].val = v;
        T[x].sm = v * T[x].sz;
        T[x].mn = T[x].mx = v;
    }

    void apply_flip(int x) {
        if(x) T[x].flip = !T[x].flip;
    }

    void rotate(int x) {
        int p = T[x].p;
        int g = T[p].p;
        int d = (T[p].c[1] == x);
        if(notRoot(p)) T[g].c[T[g].c[1] == p] = x;
        T[x].p = g;
        T[p].c[d] = T[x].c[d ^ 1];
        if (T[p].c[d]) T[T[p].c[d]].p = p;
        T[x].c[d ^ 1] = p;
        T[p].p = x;
        pull(p);
        pull(x);
    }

    void splay(int x) {
        static vi stk;
        int y = x;
        stk.pb(y);
        while (notRoot(y)) {
            y = T[y].p;
            stk.pb(y);
        }
        while(!stk.empty()) {
            push(stk.back());
            stk.pop_back();
        }
        while (notRoot(x)) {
            int p = T[x].p;
            int g = T[p].p;
            if (notRoot(p)) {
                bool dx = (T[p].c[0] == x);
                bool dy = (T[g].c[0] == p);
                if (dx ^ dy) rotate(x);
                else rotate(p);
            }
            rotate(x);
        }
    }

    int access(int x) {
        int last = 0;
        for (int y = x; y; y = T[y].p) {
            splay(y);
            T[y].c[1] = last;
            pull(y);
            last = y;
        }
        splay(x);
        return last;
    }

    void makeRoot(int x) {
        access(x);
        apply_flip(x);
        push(x);
    }

	int find_root(int u) {
        access(u);
        while(true) {
            push(u);
            int c = T[u].c[0];
            if(!c) break;
            u = c;
        }
        splay(u);
        return u;
    }

    bool is_connected(int u, int v) {
        return find_root(u) == find_root(v);
    }

    bool link(int u, int v) {
        if(find_root(u) == find_root(v)) return false;
        makeRoot(u);
        T[u].p = v;
        return true;
    }

    bool cut(int u, int v) {
        makeRoot(u);
        access(v);
        if (T[v].c[0] == u) {
            T[T[v].c[0]].p = 0;
            T[v].c[0] = 0;
            pull(v);
            return true;
        }
        return false;
    }

    int get_path(int u, int v) {
        makeRoot(u);
        access(v);
        return v;
    }

    void update_path(int u, int v, ll k, int type) {
        int x = get_path(u, v);
        if(type == 1) apply_set(x, k);
        else apply_add(x, k);
        pull(x);
    }

    Node path_queries(int u, int v) {
        int x = get_path(u, v);
        return T[x];
    }

    int rt = 1;
    void assign_root(int r) {
        rt = r;
    }

    void change_parent(int x, int y) {
        if (x == lca(x, y)) return;
        cut(rt, x);
        link(x, y);
    }

    int lca(int x, int y) {
        makeRoot(rt);
        access(x);
        return access(y);
    }
	
	bool is_ancestor(int par, int child) {
        return lca(par, child) == par;
    }
};

struct node {
    int p = 0, c[2] = {0, 0}, pp = 0;
    bool flip = 0;
    int sz = 0, ssz = 0, vsz = 0;
    ll val = 0, sum = 0, lazy = 0, subsum = 0, vsum = 0;
    node() {}
    node(int x) {
        val = x; sum = x;
        sz = 1; lazy = 0;
        ssz = 1; vsz = 0;
        subsum = x; vsum = 0;
    }
};

struct LCT {
    vector<node> t;
    LCT() {}
    LCT(int n) : t(n + 1) {}

    int dir(int x, int y) { return t[x].c[1] == y; }

    void set(int x, int d, int y) {
        if (x) t[x].c[d] = y, pull(x);
        if (y) t[y].p = x;
    }

    void pull(int x) {
        if (!x) return;
        int &l = t[x].c[0], &r = t[x].c[1];
        push(l); push(r);
        t[x].sum    = t[l].sum    + t[r].sum    + t[x].val;
        t[x].sz     = t[l].sz     + t[r].sz     + 1;
        t[x].ssz    = t[l].ssz    + t[r].ssz    + t[x].vsz + 1;
        t[x].subsum = t[l].subsum + t[r].subsum + t[x].vsum + t[x].val;
    }

    void push(int x) {
        if (!x) return;
        int &l = t[x].c[0], &r = t[x].c[1];
        if (t[x].flip) {
            swap(l, r);
            if (l) t[l].flip ^= 1;
            if (r) t[r].flip ^= 1;
            t[x].flip = 0;
        }
        if (t[x].lazy) {
            t[x].val    += t[x].lazy;
            t[x].sum    += t[x].lazy * t[x].sz;
            t[x].subsum += t[x].lazy * t[x].ssz;
            t[x].vsum   += t[x].lazy * t[x].vsz;
            if (l) t[l].lazy += t[x].lazy;
            if (r) t[r].lazy += t[x].lazy;
            t[x].lazy = 0;
        }
    }

    void rotate(int x, int d) {
        int y = t[x].p, z = t[y].p, w = t[x].c[d];
        swap(t[x].pp, t[y].pp);
        set(y, !d, w);
        set(x, d, y);
        set(z, dir(z, y), x);
    }

    void splay(int x) {
        for (push(x); t[x].p;) {
            int y = t[x].p, z = t[y].p;
            push(z); push(y); push(x);
            int dx = dir(y, x), dy = dir(z, y);
            if (!z)           rotate(x, !dx);
            else if (dx == dy) rotate(y, !dx), rotate(x, !dx);
            else               rotate(x, dy), rotate(x, dx);
        }
    }

    void make_root(int u) {
        access(u);
        int l = t[u].c[0];
        t[l].flip ^= 1;
        swap(t[l].p, t[l].pp);
        t[u].vsz  += t[l].ssz;
        t[u].vsum += t[l].subsum;
        set(u, 0, 0);
    }

    int access(int _u) {
        int last = _u;
        for (int v = 0, u = _u; u; u = t[v = u].pp) {
            splay(u); splay(v);
            t[u].vsz  -= t[v].ssz;
            t[u].vsum -= t[v].subsum;
            int r = t[u].c[1];
            t[u].vsz  += t[r].ssz;
            t[u].vsum += t[r].subsum;
            t[v].pp = 0;
            swap(t[r].p, t[r].pp);
            set(u, 1, v);
            last = u;
        }
        splay(_u);
        return last;
    }

    void link(int u, int v) {
        make_root(v);
        access(u); splay(u);
        t[v].pp  = u;
        t[u].vsz += t[v].ssz;
        t[u].vsum += t[v].subsum;
    }

    void cut(int u) {
        access(u);
        assert(t[u].c[0] != 0);
        t[t[u].c[0]].p = 0;
        t[u].c[0] = 0;
        pull(u);
    }

    int get_parent(int u) {
        access(u); splay(u); push(u);
        u = t[u].c[0]; push(u);
        while (t[u].c[1]) {
            u = t[u].c[1]; push(u);
        }
        splay(u);
        return u;
    }

    int find_root(int u) {
        access(u); splay(u); push(u);
        while (t[u].c[0]) {
            u = t[u].c[0]; push(u);
        }
        splay(u);
        return u;
    }

    bool connected(int u, int v) { return find_root(u) == find_root(v); }

    int depth(int u) { access(u); splay(u); return t[u].sz; }

    int lca(int u, int v) {
        if (u == v) return u;
        if (depth(u) > depth(v)) swap(u, v);
        access(v);
        return access(u);
    }

    int is_root(int u) { return get_parent(u) == 0; }

    int component_size(int u) { return t[find_root(u)].ssz; }

    int subtree_size(int u) {
        int p = get_parent(u);
        if (p == 0) return component_size(u);
        cut(u);
        int ans = component_size(u);
        link(p, u);
        return ans;
    }

    ll component_sum(int u) { return t[find_root(u)].subsum; }

    ll subtree_sum(int u) {
        int p = get_parent(u);
        if (p == 0) return component_sum(u);
        cut(u);
        ll ans = component_sum(u);
        link(p, u);
        return ans;
    }

    ll subtree_query(int u, int root) {
        int cur = find_root(u);
        make_root(root);
        ll ans = subtree_sum(u);
        make_root(cur);
        return ans;
    }

    ll query(int u, int v) {
        int cur = find_root(u);
        make_root(u); access(v);
        ll ans = t[v].sum;
        make_root(cur);
        return ans;
    }

    void cut(int u, int v) {
        make_root(u);
        cut(v);
    }

    int comp_size(int u) {
        int z = find_root(u);
        int c = t[z].c[1];
        return t[c].ssz;
    }

    void update_at(int u, ll v) {
        access(u);          
        t[u].val += v;     
        pull(u);          
        access(u);       
    }
};

class BITSET {
public:
    using ubig = unsigned long long;
    int sz;
    vt<ubig> blocks;
    BITSET(int n) : sz(n) {
        int len = (n + 8 * (int)sizeof(ubig) - 1) / (8 * (int)sizeof(ubig));
        blocks.assign(len, 0ULL);
    }
    void set(int i) {
        int block = i / (8 * (int)sizeof(ubig));
        int offset = i % (8 * (int)sizeof(ubig));
        blocks[block] |= (1ULL << offset);
    }
    BITSET& set() {
        for (auto &blk : blocks)
            blk = ~0ULL;
        int extra = (int)blocks.size() * 8 * (int)sizeof(ubig) - sz;
        if (extra > 0) {
            ubig mask = ~0ULL >> extra;
            blocks.back() &= mask;
        }
        return *this;
    }
    BITSET& set(int l, int r) {
        if (l < 0) l = 0;
        if (r >= sz) r = sz - 1;
        if (l > r) return *this;
        const int B = 8 * sizeof(ubig);
        int startBlock = l / B;
        int endBlock   = r / B;
        int startOff   = l % B;
        int endOff     = r % B;

        if (startBlock == endBlock) {
            ubig mask = ((~0ULL >> (B - (r - l + 1))) << startOff);
            blocks[startBlock] |= mask;
        } else {
            ubig firstMask = (~0ULL << startOff);
            blocks[startBlock] |= firstMask;
            for (int b = startBlock + 1; b < endBlock; ++b)
                blocks[b] = ~0ULL;
            ubig lastMask = (~0ULL >> (B - 1 - endOff));
            blocks[endBlock] |= lastMask;
        }
        int extra = (int)blocks.size() * B - sz;
        if (extra > 0) {
            ubig tailMask = ~0ULL >> extra;
            blocks.back() &= tailMask;
        }
        return *this;
    }
    bool test(int i) const {
        int block = i / (8 * (int)sizeof(ubig));
        int offset = i % (8 * (int)sizeof(ubig));
        return (blocks[block] >> offset) & 1ULL;
    }
    BITSET& reset() {
        fill(blocks.begin(), blocks.end(), 0ULL);
        return *this;
    }
    void reset(int i) {
        int block = i / (8 * (int)sizeof(ubig));
        int offset = i % (8 * (int)sizeof(ubig));
        blocks[block] &= ~(1ULL << offset);
    }
    BITSET& reset(int l, int r) {
        if (l < 0)       l = 0;
        if (r >= sz)     r = sz - 1;
        if (l > r)       return *this;
        const int B = 8 * sizeof(ubig);
        int startBlock = l / B;
        int endBlock   = r / B;
        int startOff   = l % B;
        int endOff     = r % B;
        if (startBlock == endBlock) {
            ubig mask = ((~0ULL >> (B - (r - l + 1))) << startOff);
            blocks[startBlock] &= ~mask;
        } else {
            ubig firstMask = (~0ULL << startOff);
            blocks[startBlock] &= ~firstMask;
            for (int b = startBlock + 1; b < endBlock; ++b)
                blocks[b] = 0ULL;
            ubig lastMask = (~0ULL >> (B - 1 - endOff));
            blocks[endBlock] &= ~lastMask;
        }
        return *this;
    }
    BITSET& flip() {
        for (auto &blk : blocks)
            blk = ~blk;
        int extra = (int)blocks.size() * 8 * (int)sizeof(ubig) - sz;
        if (extra > 0) {
            ubig mask = ~0ULL >> extra;
            blocks.back() &= mask;
        }
        return *this;
    }
    void flip(int i) {
        int block = i / (8 * (int)sizeof(ubig));
        int offset = i % (8 * (int)sizeof(ubig));
        blocks[block] ^= (1ULL << offset);
    }
    int count() const {
        int cnt = 0;
        for (auto blk : blocks)
            cnt += __builtin_popcountll(blk);
        return cnt;
    }
    bool any() const {
        for (auto blk : blocks)
            if (blk != 0) return true;
        return false;
    }
    bool none() const {
        return !any();
    }
    bool all() const {
        int fullBlocks = sz / (8 * (int)sizeof(ubig));
        for (int i = 0; i < fullBlocks; i++)
            if (blocks[i] != ~0ULL) return false;
        int remaining = sz % (8 * (int)sizeof(ubig));
        if (remaining > 0) {
            ubig mask = (1ULL << remaining) - 1;
            if (blocks[fullBlocks] != mask) return false;
        }
        return true;
    }
    string to_string() const {
        string s;
        s.resize(sz);
        for (int i = 0; i < sz; i++)
            s[sz - 1 - i] = test(i) ? '1' : '0';
        return s;
    }
    BITSET& operator|=(const BITSET& other) {
        assert(blocks.size() == other.blocks.size());
        for (size_t i = 0; i < blocks.size(); i++)
            blocks[i] |= other.blocks[i];
        return *this;
    }
    BITSET& operator&=(const BITSET& other) {
        assert(blocks.size() == other.blocks.size());
        for (size_t i = 0; i < blocks.size(); i++)
            blocks[i] &= other.blocks[i];
        return *this;
    }
    BITSET& operator^=(const BITSET& other) {
        assert(blocks.size() == other.blocks.size());
        for (size_t i = 0; i < blocks.size(); i++)
            blocks[i] ^= other.blocks[i];
        int extra = (int)blocks.size() * 8 * (int)sizeof(ubig) - sz;
        if (extra > 0) {
            ubig mask = ~0ULL >> extra;
            blocks.back() &= mask;
        }
        return *this;
    }
    BITSET operator|(const BITSET& other) const {
        BITSET res(*this);
        res |= other;
        return res;
    }
    BITSET operator&(const BITSET& other) const {
        BITSET res(*this);
        res &= other;
        return res;
    }
    BITSET operator^(const BITSET& other) const {
        BITSET res(*this);
        res ^= other;
        return res;
    }
    BITSET operator~() const {
        BITSET res(*this);
        res.flip();
        return res;
    }
    BITSET& operator<<=(int shift) {
        if(shift >= sz) {
            fill(blocks.begin(), blocks.end(), 0ULL);
            return *this;
        }
        const int B = 8 * (int)sizeof(ubig);
        int blockShift = shift / B;
        int bitShift = shift % B;
        int nblocks = blocks.size();
        vector<ubig> newBlocks(nblocks, 0ULL);
        for (int i = nblocks - 1; i >= 0; i--) {
            int srcIndex = i - blockShift;
            if (srcIndex < 0) continue;
            newBlocks[i] |= blocks[srcIndex] << bitShift;
            if (bitShift > 0 && srcIndex - 1 >= 0)
                newBlocks[i] |= blocks[srcIndex - 1] >> (B - bitShift);
        }
        blocks.swap(newBlocks);
        int extra = (int)blocks.size() * B - sz;
        if (extra > 0) {
            ubig mask = ~0ULL >> extra;
            blocks.back() &= mask;
        }
        return *this;
    }
    BITSET operator<<(int shift) const {
        BITSET res(*this);
        res <<= shift;
        return res;
    }
    
    BITSET& operator>>=(int shift) {
        if (shift >= sz) {
            fill(blocks.begin(), blocks.end(), 0ULL);
            return *this;
        }
        const int B = 8 * (int)sizeof(ubig);
        int blockShift = shift / B;
        int bitShift = shift % B;
        int nblocks = blocks.size();
        vector<ubig> newBlocks(nblocks, 0ULL);
        for (int i = 0; i < nblocks; i++) {
            int srcIndex = i + blockShift;
            if (srcIndex >= nblocks) continue;
            newBlocks[i] |= blocks[srcIndex] >> bitShift;
            if (bitShift > 0 && srcIndex + 1 < nblocks)
                newBlocks[i] |= blocks[srcIndex + 1] << (B - bitShift);
        }
        blocks.swap(newBlocks);
        int extra = (int)blocks.size() * B - sz;
        if (extra > 0) {
            ubig mask = ~0ULL >> extra;
            blocks.back() &= mask;
        }
        return *this;
    }
    BITSET operator>>(int shift) const {
        BITSET res(*this);
        res >>= shift;
        return res;
    }
    
    int find_first() const {
        const int B = 8 * (int)sizeof(ubig);
        for (size_t b = 0; b < blocks.size(); b++) {
            if (blocks[b] != 0ULL) {
                int tz = __builtin_ctzll(blocks[b]);
                int pos = b * B + tz;
                if (pos < sz)
                    return pos;
                else
                    return -1;
            }
        }
        return -1;
    }
    int find_prev_set_bit(int pos) const {
        if(pos < 0) return -1;
        if(pos >= sz) pos = sz - 1;
        if(test(pos)) return pos;
        const int B = 8 * (int)sizeof(ubig);
        int block = pos / B, offset = pos % B;
        for (int b = block; b >= 0; b--) {
            ubig mask = (b == block) ? ((1ULL << offset) - 1ULL) : ~0ULL;
            ubig curr = blocks[b] & mask;
            if (curr != 0ULL) {
                int lz = __builtin_clzll(curr);
                return b * B + (B - 1 - lz);
            }
        }
        return -1;
    }

    int find_next_set_bit(int pos) const {
        if(pos < 0) pos = 0;
        if(pos < sz && test(pos)) return pos;
        const int B = 8 * (int)sizeof(ubig);
        int block = pos / B, offset = pos % B;
        ubig mask = ~((1ULL << (offset + 1)) - 1ULL);
        ubig curr = blocks[block] & mask;
        if(curr != 0ULL) {
            int tz = __builtin_ctzll(curr);
            int res = block * B + tz;
            return (res < sz ? res : -1);
        }
        for(size_t b = block + 1; b < blocks.size(); b++) {
            if(blocks[b] != 0ULL) {
                int tz = __builtin_ctzll(blocks[b]);
                int res = b * B + tz;
                return (res < sz ? res : -1);
            }
        }
        return -1;
    }
    
    bool operator==(const BITSET& other) const {
        return blocks == other.blocks;
    }
    bool operator!=(const BITSET& other) const {
        return !(*this == other);
    }
};

// Computes all possible subset sums from 0 to n that can be made using values from sizes. Runs in O(n sqrt n / 64) if
// the sum of sizes is bounded by n, and O(n^2 / 64) otherwise.
BITSET possible_subsets_knapsack(int n, const vi &sizes) {
    vi freq(n + 1); 
    for (int s : sizes) {
        if (1 <= s && s <= n) {
            freq[s]++;
        }
    }
    BITSET knapsack(n + 1);
    knapsack.set(0);
    for (int s = 1; s <= n; s++) {
        if (freq[s] >= 3) {
            int move = (freq[s] - 1) / 2;
            if (2 * s <= n) freq[2 * s] += move;
            freq[s] -= 2 * move;
        }
        for (int r = 0; r < freq[s]; r++)
            knapsack |= knapsack << s;
    }
    return knapsack;
}

class median_tree {
public:
    void insert(int num) {
        if(left.empty() || num <= left.top()) {
            left.push(num);
            left_sum += num;
        } else {
            right.push(num);
            right_sum += num;
        }
        balance();
        clear();
    }
    
    void remove(int num) {
        if(num <= left.top()) {
            left_sum -= num;
            left_removed.push(num);
        } else {
            right_sum -= num;
            right_removed.push(num);
        }
        balance();
        clear();
    }
    
    int get_median() {
        return left.top();
    }
    
    ll get_cost() {
        ll median = get_median();
        return median * (left.size() - left_removed.size()) - left_sum + right_sum - median * (right.size() - right_removed.size());
    }
    
private:
    void balance() {
        if(left.size() - left_removed.size() >= right.size() - right_removed.size() + 2) {
            right_sum += left.top();
            left_sum -= left.top();
            right.push(left.top());
            left.pop();
        }
        if(right.size() - right_removed.size() > left.size() - left_removed.size()) {
            left_sum += right.top();
            right_sum -= right.top();
            left.push(right.top());
            right.pop();
        }
    }
    
    void clear() {
        while(!left_removed.empty() && !left.empty() && left_removed.top() > left.top()) {
            right_removed.push(left_removed.top());
            left_removed.pop();
        }
        while(!right_removed.empty() && !left.empty() && right_removed.top() < left.top()) {
            left_removed.push(right_removed.top());
            right_removed.pop();
        }
        while(!left_removed.empty() && !left.empty() && left.top() == left_removed.top()) {
            left.pop();
            left_removed.pop();
        }
        while(!right_removed.empty() && !right.empty() && right.top() == right_removed.top()) {
            right.pop();
            right_removed.pop();
        }
    }
    
    max_heap<int> left, left_removed;
    min_heap<int> right, right_removed;
    ll left_sum = 0, right_sum = 0;
};

template<typename T>
struct Mat {
    int R, C;
    vt<vt<T>> a;
    T DEFAULT; 

    Mat(const vt<vt<T>>& m, T _DEFAULT = 0) : R((int)m.size()), C(m.empty() ? 0 : (int)m[0].size()), a(m), DEFAULT(_DEFAULT) {}

    Mat(int _R, int _C, T _DEFAULT = 0) : R(_R), C(_C), DEFAULT(_DEFAULT), a(R, vt<T>(C, _DEFAULT)) {}

    static Mat identity(int n, T _DEFAULT) {
        Mat I(n, n, _DEFAULT);
        for (int i = 0; i < n; i++)
            I.a[i][i] = T(1);
        return I;
    }

    Mat operator*(const Mat& o) const {
        Mat r(R, o.C, DEFAULT);
        for (int i = 0; i < R; i++) {
            for (int k = 0; k < C; k++) {
                T v = a[i][k];
                if(v == DEFAULT) continue;
                for (int j = 0; j < o.C; j++)
                    r.a[i][j] = r.a[i][j] + v * o.a[k][j];
            }
        }
        return r;
    }

    Mat pow(ll e) const {
        Mat res = identity(R, DEFAULT), base = *this;
        while (e > 0) {
            if (e & 1) res = res * base;
            base = base * base;
            e >>= 1;
        }
        return res;
    }

    friend ostream& operator<<(ostream& os, const Mat& M) {
        for (int i = 0; i < M.R; i++) {
            for (int j = 0; j < M.C; j++) {
                os << M.a[i][j];
                if (j + 1 < M.C) os << ' ';
            }
            if (i + 1 < M.R) os << '\n';
        }
        return os;
    }
};

template<typename T>
struct wavelet_tree { // one base index
    int lo, hi;
    wavelet_tree *l, *r;
    int *b, *c;

    wavelet_tree() : lo(1), hi(0), l(nullptr), r(nullptr) {}

    void init(int *from, int *to, int x, int y) {
        lo = x, hi = y;
        if(from >= to) return;
        int mid = (lo + hi) >> 1;
        auto f = [mid](int x) { return x <= mid; };
        int n = to - from;
        b.rsz(n + 2);
        c.rsz(n + 2);
        b[0] = 0; c[0] = 0;
        for (int i = 0; i < n; i++) {
            b[i + 1] = b[i] + f(from[i]);
            c[i + 1] = c[i] + from[i];
        }
        if (hi == lo) return;
        auto pivot = stable_partition(from, to, f);
        l = new wavelet_tree();
        l->init(from, pivot, lo, mid);
        r = new wavelet_tree();
        r->init(pivot, to, mid + 1, hi);
    }

    void init(vt<T> &v) {
        reset();
        if(v.empty()) return;
        int L = MIN(v);
        int R = MAX(v);
        init(v.data(), v.data() + v.size(), L, R);
    }

    int get_kth(int l_idx, int r_idx, T k) {
        int N = (int)b.size() - 2;
        l_idx = max(1, l_idx);
        r_idx = min(r_idx, N);
        if(l_idx > r_idx) return 0;
        if(lo == hi) return lo;
        int inLeft = b[r_idx] - b[l_idx - 1], lb = b[l_idx - 1], rb = b[r_idx];
        if(k <= inLeft) return this->l->get_kth(lb + 1, rb, k);
        return this->r->get_kth(l_idx - lb, r_idx - rb, k - inLeft);
    }

    int count_less_or_equal_to(int l_idx, int r_idx, T k) {
        int N = (int)b.size() - 2;
        l_idx = max(1, l_idx);
        r_idx = min(r_idx, N);
        if(l_idx > r_idx || k < lo) return 0;
        if (hi <= k) return r_idx - l_idx + 1;
        int mid = (lo + hi) >> 1;
        int lb = b[l_idx - 1], rb = b[r_idx];
        if (k <= mid) return l->count_less_or_equal_to(lb + 1, rb, k);
        int leftCount = rb - lb;
        return leftCount + r->count_less_or_equal_to(l_idx - lb, r_idx - rb, k);
    }

    int count_equal_to(int l_idx, int r_idx, T k) {
        int N = (int)b.size() - 2;
        l_idx = max(1, l_idx);
        r_idx = min(r_idx, N);
        if(l_idx > r_idx || k < lo || k > hi) return 0;
        if(lo == hi) return r_idx - l_idx + 1;
        int lb = b[l_idx - 1], rb = b[r_idx];
        int mid = (lo + hi) >> 1;
        if(k <= mid) return this->l->count_equal_to(lb + 1, rb, k);
        return this->r->count_equal_to(l_idx - lb, r_idx - rb, k);
    }

    ll sum_less_or_equal_to(int l_idx, int r_idx, T k) {
        int N = (int)b.size() - 2;
        l_idx = max(1, l_idx);
        r_idx = min(r_idx, N);
        if(l_idx > r_idx || k < lo) return 0;
        if(hi <= k) return c[r_idx] - c[l_idx - 1];
        int lb = b[l_idx - 1], rb = b[r_idx];
        return this->l->sum_less_or_equal_to(lb + 1, rb, k) + this->r->sum_less_or_equal_to(l_idx - lb, r_idx - rb, k);
    }

    ~wavelet_tree() {
        delete l;
        delete r;
    }

    void reset() {
        if (l) {
            l->reset();
            delete l;
            l = nullptr;
        }
        if (r) {
            r->reset();
            delete r;
            r = nullptr;
        }
        b.clear();
        c.clear();
        lo = 1;
        hi = 0;
    }
};

struct square_root_decomp {
    int block_size;
    vi a;
    vll blocks;
    vll lazy;
    int n;
    
    square_root_decomp(const vi& arr) : a(arr), n(arr.size()) {
        block_size = sqrt(n);
        int num_blocks = (n + block_size - 1) / block_size;
        blocks.assign(num_blocks, 0);
        lazy.assign(num_blocks, 0);
        for (int i = 0; i < n; i++) {
            blocks[i / block_size] += a[i];
        }
    }
    
    int start_id(int i) {
        return (i / block_size) * block_size;
    }
    
    int end_id(int i) {
        return min(((i / block_size) + 1) * block_size - 1, n - 1);
    }
    
    int id(int i) {
        return i / block_size;
    }
    
    void update(int i, int x) {
        int b = id(i);
        if (lazy[b] != 0) {
            for (int j = start_id(i); j <= end_id(i); j++) {
                a[j] += lazy[b];
            }
            lazy[b] = 0;
        }
        if (a[i] == x) return;
        blocks[b] -= a[i];
        a[i] = x;
        blocks[b] += a[i];
    }
    
    ll get(int r) {
        ll res = 0;
        int b = id(r);
        for (int i = 0; i < b; i++) {
            res += blocks[i];
        }
        for (int i = b * block_size; i <= r; i++) {
            res += a[i] + lazy[b];
        }
        return res;
    }
    
    ll queries_range(int l, int r) {
        if(l == 0) return get(r);
        return get(r) - get(l - 1);
    }
    
    void update_range(int l, int r, ll x) {
        int startBlock = id(l);
        int endBlock = id(r);
        if(startBlock == endBlock) {
            if(lazy[startBlock] != 0) {
                for (int i = start_id(l); i <= end_id(l); i++) {
                    a[i] += lazy[startBlock];
                }
                lazy[startBlock] = 0;
            }
            for (int i = l; i <= r; i++) {
                a[i] += x;
                blocks[startBlock] += x;
            }
            return;
        }
        if(lazy[startBlock] != 0) {
            for (int i = start_id(l); i <= end_id(l); i++) {
                a[i] += lazy[startBlock];
            }
            lazy[startBlock] = 0;
        }
        for (int i = l; i <= end_id(l); i++) {
            a[i] += x;
            blocks[startBlock] += x;
        }
        if(lazy[endBlock] != 0) {
            for (int i = start_id(r); i <= end_id(r); i++) {
                a[i] += lazy[endBlock];
            }
            lazy[endBlock] = 0;
        }
        for (int i = start_id(r); i <= r; i++) {
            a[i] += x;
            blocks[endBlock] += x;
        }
        for (int block = startBlock + 1; block <= endBlock - 1; block++) {
            lazy[block] += x;
            int blockStart = block * block_size;
            int blockEnd = min((block + 1) * block_size - 1, n - 1);
            blocks[block] += x * (blockEnd - blockStart + 1);
        }
    }
    
    ll queries_at(int i) {
        return a[i] + lazy[id(i)];
    }
};
