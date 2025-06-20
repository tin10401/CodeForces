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
//            for(int i = 0; i < HASH_COUNT; i++) {
//                pref[i] = suff[i] = key;
//            }
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

struct TreapNode {
    int pri, size, reverse, flip;
    int key;
    ll ans, lazy_add, lazy_set;
    ll pref[2], suff[2];
    int left;
    int right;
    TreapNode(int key = 0) : reverse(0), key(key), ans(key), lazy_add(0), lazy_set(-INF), flip(0), pri(rand()), size(1), left(0), right(0) { 
//            for(int i = 0; i < HASH_COUNT; i++) {
//                pref[i] = suff[i] = key;
//            }
    }
    bool empty() { return flip == 0 && lazy_add == 0 && reverse == 0 && lazy_set == -INF; }
    void reset() { flip = lazy_add = reverse = 0; lazy_set = -INF; }
};

static TreapNode nodes[MX];
int ptr = 0;
int new_node(int key) {
    int node = ++ptr;
    nodes[node] = TreapNode(key);
    return node;
}
template<typename T>
class Treap {
private:
    int root;

    int get_size(int treap) {
        return !treap ? 0 : nodes[treap].size;
    }
    
    T get_ans(int treap) {
        return !treap ? 0 : nodes[treap].ans;
    }
    
    ll get_pref(int treap, int i) {
        return !treap ? 0 : (nodes[treap].reverse ? nodes[treap].suff[i] : nodes[treap].pref[i]);
    }

    ll get_suff(int treap, int i) {
        return !treap ? 0 : (nodes[treap].reverse ? nodes[treap].pref[i] : nodes[treap].suff[i]);
    }
    
	void unite(int treap) {  
        if(!treap) return;
        nodes[treap].size = get_size(nodes[treap].left) + get_size(nodes[treap].right) + 1;
//        for(int i = 0; i < HASH_COUNT; i++) {
//            nodes[treap].pref[i] = (get_pref(nodes[treap].right, i) + (p[i][get_size(nodes[treap].right)] * nodes[treap].key) % mod[i] + (get_pref(nodes[treap].left, i) * p[i][get_size(nodes[treap].right) + 1]) % mod[i]) % mod[i];
//            nodes[treap].suff[i] = (get_suff(nodes[treap].left, i) + (p[i][get_size(nodes[treap].left)] * nodes[treap].key) % mod[i] + (get_suff(nodes[treap].right, i) * p[i][get_size(nodes[treap].left) + 1]) % mod[i]) % mod[i];
//        }
    }

    void push(int treap) {  
        if(!treap || nodes[treap].empty()) return;
        if(nodes[treap].reverse) {
            swap(nodes[treap].left, nodes[treap].right);
            auto& L = nodes[treap].left;
            auto& R = nodes[treap].right;
            if(L) nodes[L].reverse ^= 1;
            if(R) nodes[R].reverse ^= 1;
        }
        if(nodes[treap].flip) {
            auto&L = nodes[treap].left;
            auto&R = nodes[treap].right;
            if(L) nodes[L].flip ^= 1;
            if(R) nodes[R].flip ^= 1;
        }
        if(nodes[treap].lazy_set != -INF) {
            nodes[treap].key = nodes[treap].lazy_set;
            auto& L = nodes[treap].left;
            auto&R = nodes[treap].right;
            if(L) nodes[L].lazy_set = nodes[treap].lazy_set;
            if(R) nodes[R].lazy_set = nodes[treap].lazy_set;
            nodes[treap].reset();
            unite(treap);
            return;
        }
        if(nodes[treap].lazy_add) {
            nodes[treap].key += nodes[treap].lazy_add;
            auto&L = nodes[treap].left;
            auto&R = nodes[treap].right;
            if(L) nodes[L].lazy_add += nodes[treap].lazy_add;
            if(R) nodes[R].lazy_add += nodes[treap].lazy_add;
        }
        nodes[treap].reset();
        unite(treap);
    }

    void split(int treap, int& left, int& right, int k) {
        if(!treap) {
            left = right = 0;
            return;
        }
        push(treap);
        if((by_value ? (nodes[treap].key > k) : (get_size(nodes[treap].left) >= k))) { 
            split(nodes[treap].left, left, nodes[treap].left, k);
            right = treap;
        } else {
            split(nodes[treap].right, nodes[treap].right, right, k - (by_value ? 0 : (get_size(nodes[treap].left) + 1)));
            left = treap;
        }
        unite(treap);
    }

	void merge(int& treap, int left, int right) {
        if(!left || !right) {
            treap = left ? left : right;
            return;
        }
        if(nodes[left].pri < nodes[right].pri) {
            push(left);
            merge(nodes[left].right, nodes[left].right, right);
            treap = left;
        } else {
            push(right);
            merge(nodes[right].left, left, nodes[right].left);
            treap = right;
        }
        unite(treap);
    }
public:
    bool by_value;
    Treap(bool by_value) : root(0), by_value(by_value) {} // all one base indexing
    
    void insert(T key) { 
        if(!by_value) {
            merge(root, root, new_node(key));
            return;
        }
        int A = 0;
        split(root, root, A, key - 1);
        merge(root, root, new_node(key));
        merge(root, root, A);
    }
    
	void insert_at(int k, T x) {
        if(get_size(root) < k) {
            insert(x);
            return;
        }
        int A = 0;
        split(root, root, A, k - 1);
        merge(root, root, new_node(x));
        merge(root, root, A);
    }

    void split_and_insert_at(int l, int r, int k) {
        // split s[l, r], concatnate t = s[1, l - 1] + s[r + 1, n], then insert s[i, j] at kth position of the t string
        int A = 0, B = 0;
        split(root, root, A, l - 1);
        split(A, A, B, r - l + 1);
        merge(root, root, B);
        if(get_size(root) < k) {
            merge(root, root, A);
            return;
        }
        split(root, root, B, k);
        merge(root, root, A);
        merge(root, root, B);
    }

	bool is_palindrome(int l, int r) { // https://csacademy.com/contest/archive/task/strings/statement/
        int L = 0, M = 0, R = 0;
        split(root, L, M, l - 1);
        split(M, M, R, r - l + 1);
        push(M);  
        bool ok = (nodes[M].pref[0] == nodes[M].suff[0]) && (nodes[M].pref[1] == nodes[M].suff[1]);
        merge(M, M, R);
        merge(root, L, M);
        return ok;
    }

	void split_and_apply(int l, int r, T k = 0) { 
        if(by_value) {
            int A = 0, B = 0;
            split(root, root, A, l - 1); 
            if(A) { 
                push(A);
            }
            split(A, A, B, l - 1);
            root = merge_treap(root, A);
            merge(root, root, B);
            return;
        }
        int A = 0, B = 0;
        split(root, root, A, l - 1); 
        split(A, A, B, r - l + 1);
        if(A) { 
            push(A);
        }
        merge(root, root, A);
        merge(root, root, B);
    }

    void split_and_swap(int k) {
        if(k == 0 || k == get_size(root)) return; 
        int A = 0, B = 0, C = 0;
        split(root, root, A, k);
        if(!A) return;
        merge(root, A, root);
    }

    void shift_right(int l, int r) { // [1 2 3 4] -> [4 1 2 3] and off_set by 1
        r = r - l + 1;
        int A = 0, B = 0, C = 0;
        split(root, root, A, l - 1);
        split(A, A, B, r);
        split(A, A, C, r - 1);
        merge(root, root, C);
        merge(root, root, A);
        merge(root, root, B);
    }

    void erase_at(int k) { 
        int A = 0, B = 0;
        split(root, root, A, k - 1);
        split(A, A, B, 1);
        merge(root, root, B);
    }
	
	void update_at(int k, T x) {
        if(get_size(root) < k) return;
        int A = 0, B = 0;
        split(root, root, A, k - 1);
        split(A, A, B, 1);
        A = new_node(x);
        merge(root, root, A);
        merge(root, root, B);
    }

    T queries_at(int k) {  
        int A = 0, B = 0, C = 0;
        split(root, A, B, k - 1);
        split(B, B, C, 1);  
        T ans = nodes[B].key;
        merge(root, A, B);  
        merge(root, root, C);
        return ans;
    }
	
	T queries_range(int l, int r) {
        int A = 0, B = 0;
        split(root, root, A, l - 1);
        split(A, A, B, (!by_value ? r - l + 1 : r));
        T res = get_ans(A);
        merge(root, root, A);
        merge(root, root, B);
        return res == INF ? -1 : res;
    }

	int erase_range(int l, int r) {
        int A = 0, B = 0;
        split(root, root, A, l - 1);
        split(A, A, B, r);
        merge(root, root, B);
        return A;
    }

	int merge_treap(int A, int B) {
        if(!B) return A;
        if(!A) return B;
        push(B);
        A = merge_treap(A, nodes[B].left);
        A = merge_treap(A, nodes[B].right);
        nodes[B].left = nodes[B].right = 0;
        int L = 0, R = 0;
        split(A, L, R, nodes[B].key);
        merge(L, L, B);
        merge(A, L, R);
        unite(A);
        return A;
    }

    void merge_treap(int other) {
        root = merge_treap(root, other);
    }
    
    void reset() {
        for(int i = 0; i <= ptr; i++) {
            nodes[i] = TreapNode(0);
        }
        ptr = 0;
        root = 0;
    }

    void print() {  
        print(root);
        cout << endl;
    }
	
    void print(int treap) {  
        push(treap);
        if(!treap) return;
        print(nodes[treap].left); 
        cout << char(nodes[treap].key + 'a');
        print(nodes[treap].right);
    }
	
	void print_substring(int pos, int len) { // 1 base index
        int A = 0, B = 0;
        split(root, root, A, pos - 1);
        split(A, A, B, len);
        print(A);
        cout << endl;
        merge(root, root, A);
        merge(root, root, B);
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
		if(l > r) return;
        update_at(l, val), update_at(r + 1, -val);
    }
	
	inline void reset() {
		root.assign(n, DEFAULT);
	}

	ll select(ll k) {
        ll pos = -1;
        T acc = DEFAULT;
        for(ll bit = 1LL << N; bit > 0; bit >>= 1) {
            ll np = pos + bit;
            if(np < n) {
                T cand = acc + root[np];
                if(cand.cnt < k) {
                    acc = cand;
                    pos = np;
                }
            }
        }
        return pos + 1;
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

    void update_coord_query_range(int l, int r, int low, int high) {
        add_coord(l - 1, low - 1, false);
        add_coord(l - 1, high, false);
        add_coord(r, low - 1, false);
        add_coord(r, high, false);
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
        return left + right;
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
		// don't forget the sorting, you might accidentally remove it
        auto cmp = [&](const ar(3)& a, const ar(3)& b) -> bool {    
            if(a[0] / block != b[0] / block) return a[0] / block < b[0] / block;
            int d = a[0] / block;   
            if(d & 1) return a[1] > b[1];
            return a[1] < b[1];
        }; sort(all(Q), cmp);
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
			while(r < qr) modify(a[++r], 1);
			while(l > ql) modify(a[--l], 1);
			while(r > qr) modify(a[r--], -1);
			while(l < ql) modify(a[l++], -1);
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
    vt<T> values;
    F compare;
    vi head;
    vt<array<unsigned,2>> masks;

    linear_rmq() {}

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

struct nd_prefix_sum { // prefix sum on multidimensional
    // example usage
    // int n, m, k; cin >> n >> m >> k;
    // int D = n * m * k;
    // vll a(D);
    // for(int i = 0, p = 0; i < n; i++) for(int j = 0; j < m; j++) for(int kk = 0; kk < k; kk++, p++) cin >> a[p];
    // nd_prefix_sum t(a, {n, m, k});
    // while(q--) {
    //      vi low, high; cin >> low >> high;
    // }
    int D;
    vi dims, stride;
    vll pref;

    nd_prefix_sum(const vll& data, const vi& dims_) {
        dims = dims_;
        D = dims.size();
        int N = 1;
        for (int x : dims) N *= x;
        stride.assign(D, 1);
        for (int i = D - 2; i >= 0; i--) {
            stride[i] = stride[i + 1] * dims[i + 1];
        }
        pref = data;
        for (int d = 0; d < D; d++) {
            for (int i = 0; i < N; i++) {
                int idx = (i / stride[d]) % dims[d];
                if (idx > 0) {
                    pref[i] += pref[i - stride[d]];
                }
            }
        }
    }

    ll get_flat(const vi& idx) const {
        int pos = 0;
        for (int i = 0; i < D; i++) {
            pos += idx[i] * stride[i];
        }
        return pref[pos];
    }

    ll query(const vi& lo, const vi& hi) const {
        ll res = 0;
        int maskN = 1 << D;
        vi idx(D);
        for (int mask = 0; mask < maskN; mask++) {
            int bits = pct(mask);
            ll sign = (bits % 2 ? -1 : 1);
            bool ok = true;
            for (int d = 0; d < D; d++) {
                idx[d] = (mask & (1 << d)) ? lo[d] - 1 : hi[d];
                if (idx[d] < 0) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            res += sign * get_flat(idx);
        }
        return res;
    }
};

struct interval_solver {
    vpii a;
    vi contain, is_contained, intersect;
    vvi graph;
    vi dp;
    int n;
    interval_solver(const vpii& a) : a(a), n(a.size()) {
        // all inclusive(include itself), -1 if needed
        contain.rsz(n); // how many segments does the ith contained
        is_contained.rsz(n); // how many segment does the ith be in
        intersect.rsz(n); // how many segment does the ith intersect with
        graph.rsz(n + 1);
        solve_contain();
        solve_is_contained();
        solve_intersect();
    }

    static bool cmp(const ar(3)& a, const ar(3)& b) { // [l, r, id]
        if(a[0] != b[0]) return a[0] < b[0];
        return a[1] > b[1];
    }

    void solve_is_contained() {
        var(3) A;
        for(int i = 0; i < n; i++) {
            A.pb({a[i].ff, a[i].ss, i});
        }
        sort(all(A), cmp);
        ordered_set<pii> s;
        for(auto& [l, r, id] : A) {
            s.insert({r, id});
            is_contained[id] = s.size() - s.order_of_key(MP(r, -1));
        }
    }
    
    void solve_contain() {
        var(3) A;
        for(int i = 0; i < n; i++) {
            A.pb({a[i].ff, a[i].ss, i});
        }
        sort(all(A), cmp);
        ordered_set<pii> s;
        for(int i = n - 1; i >= 0; i--) {
            auto& [l, r, id] = A[i];
            s.insert({r, id});
            contain[id] = s.order_of_key(MP(r + 1, -1)); 
        }
    }

    void solve_intersect() {
        var(3) A;
        for(int i = 0; i < n; i++) {
            A.pb({a[i].ff, a[i].ss, i});
        }
        sort(all(A), cmp);
        min_heap<int> q;
        for(int i = 0; i < n; i++) {
            auto& [l, r, id] = A[i];
            while(!q.empty() && q.top() < l) q.pop();
            int left = i, right = n - 1, extra = 0;
            while(left <= right) {
                int middle = midPoint;
                if(A[middle][0] <= r) extra = middle - i + 1, left = middle + 1;
                else right = middle - 1;
            }
            intersect[i] = q.size() + extra;
            q.push(r);
        }
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
    BITSET extract(int l, int r) const { // turn off every bit not in range [l, r]
        BITSET res(sz);
        if (l < 0) l = 0;
        if (r >= sz) r = sz - 1;
        if (l > r) return res;
        const int B = 8 * sizeof(ubig);

        int startBlock = l / B, endBlock = r / B;
        int startOff = l % B, endOff = r % B;

        if (startBlock == endBlock) {
            ubig mask = ((~0ULL >> (B - (r - l + 1))) << startOff);
            res.blocks[startBlock] = blocks[startBlock] & mask;
        } else {
            ubig firstMask = ~0ULL << startOff;
            res.blocks[startBlock] = blocks[startBlock] & firstMask;
            for (int b = startBlock + 1; b < endBlock; ++b)
                res.blocks[b] = blocks[b];
            ubig lastMask = (~0ULL >> (B - 1 - endOff));
            res.blocks[endBlock] = blocks[endBlock] & lastMask;
        }
        int extra = (int)blocks.size() * B - sz;
        if (extra > 0) {
            ubig tailMask = ~0ULL >> extra;
            res.blocks.back() &= tailMask;
        }
        return res;
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

vi min_knapsack(int n, const vi& a) { // return a vector which a[i] is min_element to reach sum_i, sum is bounded by n, giving n * sqrt(n) * log(n) sometime faster
    // https://codeforces.com/contest/95/problem/E
    vi count(n + 1, 0);
    for(int sz : a) {
        if(sz > 0 && sz <= n) count[sz]++;
    }

    vi dp(n + 1, inf), next_dp;
    dp[0] = 0;

    for(int s = 1; s <= n; ++s) {
        int cnt = count[s];
        if(cnt <= 0) continue;

        for(int r = 0; r < s; ++r) {
            deque<pii> dq; // (q, value)
            for(int j = r, q = 0; j <= n; j += s, ++q) {
                int val = dp[j] - q;
                while(!dq.empty() && dq.front().ff < q - cnt) dq.pop_front();
                while(!dq.empty() && dq.back().ss >= val) dq.pop_back();
                dq.emplace_back(q, val);
                dp[j] = min(dp[j], dq.front().ss + q);
            }
        }
    }
    return dp;
}

template<typename T>
vector<T> sum_knapsack(int n, const vi& a) {
    // return the # of subset sum to each dp[i]
    // nqrt(n) 
    vi cnt(n + 1, 0);
    for(int x : a) {
        if(x > 0 && x <= n) cnt[x]++;
    }
    vt<T> dp(n + 1, T(0)), ndp(n + 1, T(0));
    dp[0] = T(1);
    for(int v = 1; v <= n; ++v) {
        int c = cnt[v];
        if(c == 0) continue;
        fill(ndp.begin(), ndp.end(), T(0));
        for(int r0 = 0; r0 < v; ++r0) {
            T window_sum = T(0);
            int maxK = (n - r0) / v;
            for(int k = 0; k <= maxK; ++k) {
                int j = r0 + k * v;
                window_sum = window_sum + dp[j];
                if(k - (c + 1) >= 0) {
                    int jrem = r0 + (k - (c + 1)) * v;
                    window_sum = window_sum - dp[jrem];
                }
                ndp[j] = window_sum;
            }
        }
        dp.swap(ndp);
    }
    int zero = count(all(a), 0);
    for(auto& x : dp) {
        x *= (zero + 1);
    }
    return dp;
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
struct range_unique { // determine if a[l, r] contain all unique value
    vi safe;
    int n;
    range_unique(const vt<T>& a) : n(a.size()), safe(a.size()) {
        map<int, int> last;
        for(int i = 0, l = -1; i < n; i++) {
            if(last.count(a[i])) l = max(l, last[a[i]] + 1);
            safe[i] = l;
            last[a[i]] = i;
        }
    }

    bool all_unique(int l, int r) {
        return safe[r] <= l;
    }
};

struct bracket {
    vi prefix;
    int n;
    linear_rmq<int> rq;
    vi right_most; // longest balance bracket sequence starting at this index
    bracket() {}

    bracket(const string& s) {
        n = s.size();
        prefix.rsz(n + 1);
        right_most.rsz(n, -1);
        for(int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + (s[i] == '(' ? 1 : -1);
        }
        rq = linear_rmq<int>(prefix, [](const int& a, const int& b) {return a < b;});
        stack<int> st;
        for(int i = 0; i < n; i++) {
            if(s[i] == '(') {
                st.push(i);
                continue;
            }
            if(!st.empty() && s[st.top()] == '(') {
                right_most[st.top()] = i;
                st.pop();
            }
            else st.push(i);
        }
        for(int i = n - 1; i >= 0; i--) {
            int r = right_most[i];
            if(r == -1) continue;
            if(r + 1 < n && right_most[r + 1] != -1) right_most[i] = right_most[r + 1];
        }
    }

    bool is_balanced(int l, int r) {
        return rq.query(l + 1, r + 1) - prefix[l] >= 0 && prefix[r + 1] - prefix[l] == 0;
    }

    int longest_balance_bracket_starting_at(int i) {
        if(right_most[i] == -1) return -inf;
        return right_most[i] - i + 1;
    }

    int max_balanced_substring_len(vector<string>& A) { // given an array of bracket string, find the max subarray over all permutation
                                                        // TC : A.size() * sum(str)
        // https://open.kattis.com/problems/piecesofparentheses
        struct Piece { int diff, mn, len; };
        int n = A.size();
        vt<Piece> pieces;
        int sumPos = 0;
        for(auto& s : A) {
            int bal = 0, m = 0;
            for(char c : s) {
                bal += (c == '(' ? 1 : -1);
                m = min(m, bal);
            }
            pieces.pb({bal, m, (int)s.size()});
            if(bal > 0) sumPos += bal;
        }
        sort(all(pieces), [&](auto &a, auto &b){
                if(a.mn != b.mn) return a.mn > b.mn;
                if(a.diff != b.diff) return a.diff > b.diff;
                return a.len < b.len;
                });
        int M = sumPos;
        vi dp(M + 1, -1), highest(M + 1, -1);
        dp[0] = 0;
        highest[0] = 0;
        for(auto &p : pieces) {
            int d = p.diff, m = p.mn, t = p.len;
            if(d >= 0) {
                for(int j = M; j >= d; j--) {
                    int prev = j - d;
                    if(dp[prev] < 0) continue;
                    if(highest[prev] + m < 0) continue;
                    int candLen = dp[prev] + t;
                    int candHigh = highest[prev] + d;
                    if(candLen > dp[j]) {
                        dp[j] = candLen;
                        highest[j] = candHigh;
                    } else if(candLen == dp[j] && candHigh > highest[j]) {
                        highest[j] = candHigh;
                    }
                }
            } else {
                for(int j = 0; j <= M + d; j++) {
                    int prev = j - d;
                    if(dp[prev] < 0) continue;
                    if(highest[prev] + m < 0) continue;
                    int candLen = dp[prev] + t;
                    int candHigh = highest[prev];
                    if(candLen > dp[j]) {
                        dp[j] = candLen;
                        highest[j] = candHigh;
                    } else if(candLen == dp[j] && candHigh > highest[j]) {
                        highest[j] = candHigh;
                    }
                }
            }
        }
        return max(0, dp[0]);
    }
};

template<typename T>
struct wavelet_tree { // one base index
    int lo, hi;
    wavelet_tree *l, *r;
    vi b, c;

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

template <typename T, bool smallest_tie = false>
struct Static_Range_Mode_Query {
    using u32 = uint32_t;
    vt<T> V;
    vvpii mode;
    vi A, occur, start, pos;
    int bk;

    Static_Range_Mode_Query() = default;
    Static_Range_Mode_Query(const vt<T>& Vec) { init(Vec); }

    void init(const vt<T>& Vec) {
        int n = Vec.size();
        V = Vec;
        srtU(V);
        bk = sqrt(n);
        A.rsz(n);
        for(int i = 0; i < n; i++) A[i] = int(lb(all(V), Vec[i]) - V.begin());
        occur.rsz(n);
        pos.rsz(n);
        start.rsz(V.size() + 1);
        for (int i = 0; i < n; i++) occur[i] = start[A[i]]++;
        inclusive_scan(all(start), start.begin());
        for (int i = n; i--; ) pos[--start[A[i]]] = i;
        int blocks = (n + bk - 1) / bk;
        mode.assign(blocks, vt<pii>(blocks));
        vi cnt(V.size());
        for(int i = 0; i + bk <= n; i += bk) {
            int l = i / bk;
            fill(cnt.begin(), cnt.end(), 0);
            pii now_mode = {0, 0};
            for(int j = i; j + bk <= n; j += bk) {
                int r = j / bk;
                for(int k = j; k < j + bk; k++) {
                    int v = A[k];
                    int c = ++cnt[v];
                    if(c > now_mode.first || (c == now_mode.first && (smallest_tie ? v < now_mode.second : v > now_mode.second))) {
                        now_mode = {c, v};
                    }
                }
                mode[l][r] = now_mode;
            }
        }
    }

    pair<T,int> query(int l, int r) const {
        if(l > r) return {T(), 0};
        int r_excl = r + 1;
        int lb = (l == 0 ? 0 : (l - 1) / bk + 1);
        int rb = r_excl / bk;
        if(lb >= rb) {
            int freq = 0;
            int value = 0;
            for(int i = l; i < r_excl; i++) {
                int j = occur[i] + start[A[i]];
                while(j + freq < start[A[i] + 1] && pos[j + freq] < r_excl) {
                    freq++;
                    value = A[i];
                }
            }
            return {V[value], freq};
        }
        auto [freq, value] = mode[lb][rb - 1];
        for(int i = l; i < lb * bk; i++) {
            int j = occur[i] + start[A[i]];
            while(j + freq < start[A[i] + 1] && pos[j + freq] < r_excl) {
                freq++;
                value = A[i];
            }
        }
        for(int i = rb * bk; i < r_excl; i++) {
            int j = occur[i] + start[A[i]];
            while(j - freq >= start[A[i]] && pos[j - freq] >= l) {
                freq++;
                value = A[i];
            }
        }
        return {V[value], freq};
    }
};

struct RangeSorter {
    // use 1 base index array
    // works for permutation
    int POOLSZ;
    int free_top;
    var(2) ch;
    vi rots, rs, typ, cnt, free_list;
    set<int> runs;
    int n;

    RangeSorter(int _n, const vi& a)
        : POOLSZ(_n * ((int)log2(_n) + 3)), free_list(POOLSZ), ch(POOLSZ), cnt(POOLSZ),
          rots(_n + 2), rs(_n + 2), typ(_n + 2), n(_n)
    {
        free_top = POOLSZ - 1;
        for(int i = 1; i < POOLSZ; i++) free_list[i] = i;
        runs.clear();
        for(int i = 1; i <= n; i++) {
            rs[i] = i;
            typ[i] = false;
            runs.insert(i);
            build_node(rots[i], 1, n, a[i]);
        }
    }

    int alloc_node() {
        int x = free_list[free_top--];
        ch[x][0] = ch[x][1] = cnt[x] = 0;
        return x;
    }

    void build_node(int &T, int L, int R, int p) {
        T = alloc_node();
        cnt[T] = 1;
        if(L == R) return;
        int M = (L + R) >> 1;
        if(p <= M) build_node(ch[T][0], L, M, p);
        else build_node(ch[T][1], M + 1, R, p);
    }

    int merge_tree(int t1, int t2) {
        if(!t1 || !t2) return t1 ^ t2;
        ch[t1][0] = merge_tree(ch[t1][0], ch[t2][0]);
        ch[t1][1] = merge_tree(ch[t1][1], ch[t2][1]);
        cnt[t1] += cnt[t2];
        free_list[++free_top] = t2;
        return t1;
    }

    void split_tree(int t1, int &t2, int k) {
        t2 = alloc_node();
        int ls = cnt[ch[t1][0]];
        if(k > ls) split_tree(ch[t1][1], ch[t2][1], k - ls);
        else swap(ch[t1][1], ch[t2][1]);
        if(k < ls) split_tree(ch[t1][0], ch[t2][0], k);
        cnt[t2] = cnt[t1] - k;
        cnt[t1] = k;
    }

    int ask_kth(int T, int L, int R, int k) {
        if(L == R) return L;
        int ls = cnt[ch[T][0]];
        int M = (L + R) >> 1;
        if(k <= ls) return ask_kth(ch[T][0], L, M, k);
        else return ask_kth(ch[T][1], M + 1, R, k - ls);
    }

    void split_at(int x) {
        if(x < 1 || x > n) return;
        auto it = prev(runs.ub(x));
        int st = *it;
        if (st == x) return;
        int en = rs[st];
        int len_left = x - st;
        bool desc = typ[st];
        int root = rots[st];
        runs.erase(it);

        int left_root, right_root;
        if(!desc) {
            split_tree(root, right_root, len_left);
            left_root = root;
        } else {
            split_tree(root, left_root, (en - st + 1) - len_left);
            right_root = root;
        }
        rots[st] = left_root;
        rs[st] = x - 1;
        typ[st] = desc;
        rots[x] = right_root;
        rs[x] = en;
        typ[x] = desc;

        runs.insert(st);
        runs.insert(x);
    }

    void merge_runs(int a, int b) {
        runs.erase(b);
        rots[a] = merge_tree(rots[a], rots[b]);
        rs[a] = rs[b];
    }

    void sort_range(int L, int R, bool descending) {
        split_at(L);
        split_at(R + 1);
        auto itL = runs.lb(L);
        auto itR = runs.ub(R);
        vi to_merge;
        for(auto it = itL; it != itR; ++it)
            to_merge.pb(*it);
        int base = to_merge[0];
        for(size_t i = 1; i < to_merge.size(); i++)
            merge_runs(base, to_merge[i]);
        typ[base] = descending;
    }

    void sort_ascending(int L, int R) { sort_range(L, R, false); }
    void sort_descending(int L, int R) { sort_range(L, R, true); }

    int get(int pos) {
        auto it = prev(runs.ub(pos));
        int st = *it;
        int offset = pos - st + 1;
        bool desc = typ[st];
        int length = rs[st] - st + 1;
        if(!desc) return ask_kth(rots[st], 1, n, offset);
        else return ask_kth(rots[st], 1, n, length - offset + 1);
    }

    vi final_array() {
        vi ans(n);
        for (int i = 1; i <= n; i++)
            ans[i - 1] = get(i);
        return ans;
    }
};

template<typename T, typename F = function<T(const T&, const T&)>>
struct MonoQueue {
    // can handle anything that's associative
    // sum, xor, min, max, and, or, product, gcd
    // careful with the default
    F op;
    T e;
    stack<pair<T,T>> in, out;

    MonoQueue(T DEFAULT, F _op) : op(_op), e(DEFAULT) {}

    void push(T x) {
        T agg = in.empty() ? x : op(in.top().ss, x);
        in.emplace(x, agg);
    }

    void pop() {
        if(out.empty()) {
            while(!in.empty()) {
                T v = in.top().ff;
                in.pop();
                T agg = out.empty() ? v : op(v, out.top().ss);
                out.emplace(v, agg);
            }
        }
        if(!out.empty()) out.pop();
    }

    T query() const {
        if(in.empty() && out.empty()) return e;
        if(in.empty()) return out.top().ss;
        if(out.empty()) return in.top().ss;
        return op(in.top().ss, out.top().ss);
    }

    bool empty() const {
        return in.empty() && out.empty();
    }
};

template<int K>
struct ODT {
    // https://atcoder.jp/contests/abc237/tasks/abc237_g
    struct Node {
        int l, r, v;
        bool operator<(Node const &o) const { return l < o.l; }
    };

    set<Node> s;

    ODT(int n, const vector<int> &a) {
        for (int i = 1; i <= n; i++) s.insert({i, i, a[i]});
    }

    auto split(int pos) {
        auto it = s.lower_bound({pos, 0, 0});
        if (it != s.end() && it->l == pos) return it;
        --it;
        Node cur = *it;
        s.erase(it);
        s.insert({cur.l, pos - 1, cur.v});
        return s.insert({pos, cur.r, cur.v}).first;
    }

    void sort_increasing(int l, int r) {
        auto itr = split(r + 1), itl = split(l);
        array<int, K> cnt{};
        for (auto it = itl; it != itr; ++it) cnt[it->v] += it->r - it->l + 1;
        s.erase(itl, itr);
        int cur = l;
        for (int v = 0; v < K; v++) {
            int c = cnt[v];
            if (!c) continue;
            s.insert({cur, cur + c - 1, v});
            cur += c;
        }
    }

    void sort_descending(int l, int r) {
        auto itr = split(r + 1), itl = split(l);
        array<int, K> cnt{};
        for (auto it = itl; it != itr; ++it) cnt[it->v] += it->r - it->l + 1;
        s.erase(itl, itr);
        int cur = l;
        for (int v = K - 1; v >= 0; v--) {
            int c = cnt[v];
            if (!c) continue;
            s.insert({cur, cur + c - 1, v});
            cur += c;
        }
    }

    int get(int pos) const {
        auto it = s.upper_bound({pos, 0, 0});
        --it;
        return it->v;
    }
};

template<typename T>
struct arithmetic_prefix { // 0 index
    vt<T> a;
    vll PREFIX, prefix;
    int n;
    arithmetic_prefix(const vt<T>& a) : a(a), n(a.size()) {
        prefix.rsz(n + 1);
        PREFIX.rsz(n + 1);
        for(int i = 1; i <= n; i++) {
            prefix[i] = prefix[i - 1] + a[i - 1];
            PREFIX[i] = PREFIX[i - 1] + ((ll)a[i - 1] * i);
        }
    }

    ll query_prefix(int l, int r, bool inclusive = true) {
        l++, r++;
        ll big = PREFIX[r] - PREFIX[l - 1];
        ll small = prefix[r] - prefix[l - 1];
        return big - (inclusive ? small * (l - 1) : small * l);
    }
    
    ll query_suffix(int l, int r, bool inclusive = true) {
        l++, r++;
        ll big = PREFIX[r] - PREFIX[l - 1];
        ll small = prefix[r] - prefix[l - 1];
        ll base = inclusive ? r + 1 : r;
        return small * base - big;
    }

    ll median_split_inward(int l, int r, bool inclusive = true) { // find best point to split between [l, r]
                                                                  // the cost bring inward
        int left = l - 1, right = r;
        ll res = INF;
        while(left <= right) {
            int middle = midPoint;
            ll A = middle < l ? 0 : query_prefix(l, middle, inclusive);
            ll B = middle == r ? 0 : query_suffix(middle + 1, r, inclusive);
            res = min(res, A + B);
            if(A < B) left = middle + 1;
            else right = middle - 1;
        }
        return res;
    }

    ll get_prefix(int l, int r) {
        return prefix[r + 1] - prefix[l];
    }
    
    ll median_split_outward(int l, int r) { // find the best point to split between [l, r]
                                            // then cost is query_suffix(l, p, false) + query_prefix(p + 1, r)
        ll tot  = get_prefix(l, r);
        ll half = (tot + 1) / 2;
        int left = l, right = r;
        while(left < right) {
            int middle = midPoint;
            if(get_prefix(l, middle) < half) left = middle + 1;
            else right = middle;
        }
        int m = left;;
        ll leftCost = (m >= l) ? query_suffix(l, m, false) : 0;
        ll rightCost = (m + 1 <= r) ? query_prefix(m + 1, r, true) : 0;
        return leftCost + rightCost;
    }
};

struct Alien_trick {
    struct state {
        ll seg;
        ll val;
        state(ll val = -INF, ll seg = 0) : val(val), seg(seg) {}
    };

    Alien_trick() {}

    state cmp(const state& a, const state& b) {
        if(a.val != b.val) return a.val > b.val ? a : b;
        return a.seg < b.seg ? a : b;
    }
    
    ll run(const vi& a, ll k) {
        int n = a.size();
        auto f = [&](ll cost) -> state {
            state out(0, 0), in; // either continue a segment, infer no cost, or starting a new segment, costing cost extra
            for(int i = 0; i < n; i++) {
                state next_out = cmp(out, in), next_in;
                state op1(out.val + a[i] - cost, out.seg + 1);
                state op2(in.val + a[i], in.seg);
                next_in = cmp(op1, op2);
                swap(out, next_out);
                swap(in, next_in);
            }
            return cmp(in, out);
        };
        ll left = -1, right = INF;
        while(left + 1 < right) {
            ll middle = midPoint;
            auto now = f(middle);
            if(now.seg > k) left = middle;
            else right = middle;
        }
        auto st = f(right);
        auto res =  st.val + right * k;
        return res;
    }

    ll run(const vi& a, ll k, ll len) {
        int n = a.size();
        vll prefix(n + 1);
        for(int i = 1; i <= n; i++) {
            prefix[i] = prefix[i - 1] + a[i - 1];
        }
        auto f = [&](ll cost) -> state {
            vt<state> dp(n + 1);
            dp[0] = state(0, 0);
            for(int i = 1; i <= n; i++) {
                dp[i] = cmp(dp[i], dp[i - 1]);
                if(i >= len) {
                    int j = i - len;
                    ll cover = prefix[i] - prefix[j];
                    state now(dp[j].val + cover - cost, dp[j].seg + 1);
                    dp[i] = cmp(dp[i], now);
                }
                if(i == n) {
                    for(int j = n - len; j < n; j++) {
                        if(j < 0) continue;
                        ll cover = prefix[i] - prefix[j];
                        state now(dp[j].val + cover - cost, dp[j].seg + 1);
                        dp[i] = cmp(dp[i], now);
                    }
                }
            }
            return dp[n];
        };
        ll left = -1, right = INF;
        while(left + 1 < right) {
            ll middle = midPoint;
            auto now = f(middle);
            if(now.seg > k) left = middle;
            else right = middle;
        }
        auto st = f(right);
        auto res =  st.val + right * k;
        return res;
    }
};

struct DNC {
	// iterative impl : https://cses.fi/problemset/result/13062862/
    int n;
    vll &dp;
    vll next;
    DNC(vll &_dp) : dp(_dp), n((int)_dp.size() - 1), next(_dp.size()) { }

    template<typename Eval>
    void run(Eval eval) {
        fill(all(next), INF);
        dfs(1, n, 1, n, eval);
        swap(dp, next);
        
    }
    
    template<typename Eval>
    void dfs(int l, int r, int idl, int idr, Eval eval) {
        if(l > r) return;
        int mid = (l + r) >> 1;
        ll best = INF;
        int best_k = idl;
        for(int p = idl; p <= min(mid, idr); p++) {
            ll v = eval(p, mid);
            if(v < best) {
                best = v;
                best_k = p;
            }
        }
        next[mid] = best;
        dfs(l, mid - 1, idl, best_k, eval);
        dfs(mid + 1, r, best_k, idr, eval);
    }
};

struct range_lis_query_impl { // only works for permutation
    struct wavelet_matrix_impl {
        using uint = unsigned int;
        static constexpr int w = CHAR_BIT * sizeof(uint);

        static int popcount(uint x) {
#ifdef __GNUC__
            return __builtin_popcount(x);
#else
            static_assert(w == 32, "");
            x -= (x >> 1) & 0x55555555;
            x  = (x & 0x33333333) + ((x >> 2) & 0x33333333);
            x  = (x + (x >> 4)) & 0x0F0F0F0F;
            return (x * 0x01010101 >> 24) & 0x3F;
#endif
        }

        class bit_vector {
            struct node_type { uint bit = 0; int sum = 0; };
            vector<node_type> v;
        public:
            explicit bit_vector(uint n) : v(n / w + 1) {}
            void set(uint i) { v[i / w].bit |= uint(1) << (i % w); ++v[i / w].sum; }
            void build() { for (size_t i = 1; i < v.size(); ++i) v[i].sum += v[i - 1].sum; }
            int rank(uint i) const { return v[i / w].sum - popcount(v[i / w].bit & (~uint(0) << (i % w))); }
            int one()  const { return v.back().sum; }
        };

        class wavelet_matrix {
            template <class I> static bool test(I x, int k) { return (x & (I(1) << k)) != 0; }
            vector<bit_vector> mat;
        public:
            template <class I>
            wavelet_matrix(int bit_len, vector<I> a) : mat(bit_len, bit_vector(a.size())) {
                int n = a.size();
                vector<I> tmp; tmp.reserve(n);
                for (int p = bit_len - 1; p >= 0; --p) {
                    bit_vector &bv = mat[p];
                    auto it = a.begin();
                    for (int i = 0; i < n; ++i) {
                        if (test(a[i], p)) { bv.set(i); *it++ = a[i]; }
                        else               { tmp.push_back(a[i]); }
                    }
                    bv.build();
                    copy(tmp.begin(), tmp.end(), it);
                    tmp.clear();
                }
            }

            int count_less_than(int l, int r, ll key) const {
                int ret = r - l;
                for (int p = int(mat.size()) - 1; p >= 0; --p) {
                    const bit_vector &bv = mat[p];
                    int rl = bv.rank(l), rr = bv.rank(r);
                    if (test(key, p)) { l = rl; r = rr; }
                    else {
                        ret -= rr - rl;
                        int o = bv.one();
                        l += o - rl;
                        r += o - rr;
                    }
                }
                return ret - (r - l);
            }
        };
    };

    using wavelet_matrix = wavelet_matrix_impl::wavelet_matrix;
    using ptr  = vi::iterator;
    static constexpr int none = -1;

    static vi inverse(const vi &p) {
        int n = p.size();
        vi q(n, none);
        for (int i = 0; i < n; ++i) if (p[i] != none) q[p[i]] = i;
        return q;
    }

    static void unit_monge_dmul(int n, ptr st, ptr a, ptr b) {
        if (n == 1) { st[0] = 0; return; }

        ptr c_row = st; st += n;
        ptr c_col = st; st += n;

        auto map_fn = [&](int len, auto f, auto g) {
            ptr a_h = st + 0 * len;
            ptr a_m = st + 1 * len;
            ptr b_h = st + 2 * len;
            ptr b_m = st + 3 * len;

            auto split = [&](ptr v, ptr vh, ptr vm) {
                for (int i = 0; i < n; ++i)
                    if (f(v[i])) { *vh++ = g(v[i]); *vm++ = i; }
            };

            split(a, a_h, a_m);
            split(b, b_h, b_m);

            ptr c = st + 4 * len;
            unit_monge_dmul(len, c, a_h, b_h);

            for (int i = 0; i < len; ++i) {
                int row = a_m[i];
                int col = b_m[c[i]];
                c_row[row] = col;
                c_col[col] = row;
            }
        };

        int mid = n / 2;
        map_fn(mid,     [mid](int x){ return x <  mid; }, [](int x){ return x;       });
        map_fn(n - mid, [mid](int x){ return x >= mid; }, [mid](int x){ return x - mid; });

        struct d_itr { int delta = 0; int col = 0; } neg, pos;
        int row = n;

        auto move_right = [&](d_itr &it) {
            if (b[it.col] < mid ? c_col[it.col] >= row : c_col[it.col] < row) ++it.delta;
            ++it.col;
        };

        auto up = [&](d_itr &it) {
            if (a[row] < mid ? c_row[row] >= it.col : c_row[row] < it.col) --it.delta;
        };

        while (row) {
            while (pos.col != n) {
                d_itr t = pos;
                move_right(t);
                if (!t.delta) pos = t; else break;
            }
            --row;
            up(neg);
            up(pos);
            while (neg.delta) move_right(neg);
            if (neg.col > pos.col) c_row[row] = pos.col;
        }
    }

    static vi subunit_monge_dmul(vi a, vi b) {
        int n = a.size();
        vi a_inv = inverse(a), b_inv = inverse(b);
        swap(b, b_inv);

        vi a_map, b_map;
        for (int i = n - 1; i >= 0; --i) if (a[i] != none) { a_map.push_back(i); a[n - a_map.size()] = a[i]; }
        reverse(a_map.begin(), a_map.end());

        int cnt = 0;
        for (int i = 0; i < n; ++i) if (a_inv[i] == none) a[cnt++] = i;

        for (int i = 0; i < n; ++i) if (b[i] != none) { b[b_map.size()] = b[i]; b_map.push_back(i); }
        cnt = b_map.size();
        for (int i = 0; i < n; ++i) if (b_inv[i] == none) b[cnt++] = i;

        int stack_size = [](int m){ int ret = 0; while (m > 1) { ret += 2 * m; m = (m + 1) / 2; ret += 4 * m; } return ret + 1; }(n);

        vi c(stack_size);
        unit_monge_dmul(n, c.begin(), a.begin(), b.begin());

        vi c_pad(n, none);
        for (int i = 0; i < (int)a_map.size(); ++i) {
            int t = c[n - a_map.size() + i];
            if (t < (int)b_map.size()) c_pad[a_map[i]] = b_map[t];
        }
        return c_pad;
    }

    static vi seaweed_doubling(const vi &p) {
        int n = p.size();
        if (n == 1) return vi{none};
        int mid = n / 2;

        vi lo, hi, lo_map, hi_map;
        for (int i = 0; i < n; ++i) {
            int e = p[i];
            if (e < mid) { lo.push_back(e); lo_map.push_back(i); }
            else         { hi.push_back(e - mid); hi_map.push_back(i); }
        }

        lo = seaweed_doubling(lo);
        hi = seaweed_doubling(hi);

        vi lo_pad(n), hi_pad(n);
        iota(lo_pad.begin(), lo_pad.end(), 0);
        iota(hi_pad.begin(), hi_pad.end(), 0);

        for (int i = 0; i < mid; ++i)          lo_pad[lo_map[i]] = (lo[i] == none) ? none : lo_map[lo[i]];
        for (int i = 0; mid + i < n; ++i)     hi_pad[hi_map[i]] = (hi[i] == none) ? none : hi_map[hi[i]];

        return subunit_monge_dmul(move(lo_pad), move(hi_pad));
    }

    static bool is_permutation(const vi &p) {
        int n = p.size();
        vector<bool> used(n, false);
        for (int e : p) {
            if (e < 0 || e >= n || used[e]) return false;
            used[e] = true;
        }
        return true;
    }

    static wavelet_matrix convert(const vi &p) {
        assert(is_permutation(p));
        int n = p.size();
        vi row = n ? seaweed_doubling(vi(p.begin(), p.end())) : vi();
        for (int &e : row) if (e == none) e = n;
        int bit_len = 0; for (int t = n; t; t >>= 1) ++bit_len;
        return wavelet_matrix(bit_len, move(row));
    }

    class range_lis_query {
        int n;
        wavelet_matrix wm;
    public:
        range_lis_query() = default;
        explicit range_lis_query(const vector<int> &p) : n(p.size()), wm(convert(p)) {}
        int query(int l, int r) const {
            assert(0 <= l && l <= r && r < n);
            return (r - l + 1) - wm.count_less_than(l, n, r + 1);
        }
    };
}; using range_lis = range_lis_query_impl::range_lis_query;

struct static_range_palindrome {
    int n;
    vi a, b, c, d, d1, d2;
    wavelet_psgt oddl, oddr, evenl, evenr;

    static_range_palindrome(const string &s) {
        n = (int)s.size();
        d1.assign(n, 0);
        d2.assign(n, 0);
        build_manachers(s);

        a.rsz(n);
        b.rsz(n);
        c.rsz(n);
        d.rsz(n);
        for (int i = 0; i < n; i++) {
            a[i] = d1[i] - (i + 1);
            b[i] = d1[i] + (i + 1);
            c[i] = d2[i] - (i + 1);
            d[i] = d2[i] + (i + 1);
        }

        oddl = wavelet_psgt(a);
        oddr = wavelet_psgt(b);
        evenl = wavelet_psgt(c);
        evenr = wavelet_psgt(d);
    }

    // number of odd-length palindromes fully inside [l..r]
    ll query_odd(int l, int r) {
        return compute_odd(l, r);
    }
    // number of even-length palindromes fully inside [l..r]
    ll query_even(int l, int r) {
        return compute_even(l, r);
    }

    ll query_all(int l, int r) {
        return query_odd(l, r) + query_even(l, r);
    }

private:
    void build_manachers(const string &s) {
        for(int i = 0, L = 0, R = -1; i < n; i++) {
            int k = (i > R ? 1 : min(d1[L + R - i], R - i + 1));
            while(i - k >= 0 && i + k < n && s[i - k] == s[i + k]) k++;
            d1[i] = k--;
            if(i + k > R) { L = i - k; R = i + k; }
        }
        for(int i = 0, L = 0, R = -1; i < n; i++) {
            int k = (i > R ? 0 : min(d2[L + R - i + 1], R - i + 1));
            while(i - k - 1 >= 0 && i + k < n && s[i - k - 1] == s[i + k]) k++;
            d2[i] = k--;
            if(i + k > R) { L = i - k - 1; R = i + k; }
        }
    }

    inline ll get(int L, int R) {
        return (ll)R * (R + 1) / 2 - (ll)(L - 1) * L / 2;
    }

    ll compute_odd(int l, int r) {
        int m = (l + r) >> 1;
        int c1 = -l;
        auto n1 = oddl.query_leq(l, m, c1);
        ll less1 = n1.cnt, sum1 = n1.sm;
        ll left  = get(l + 1, m + 1) + sum1 + (ll)(m - l + 1 - less1) * c1;

        int c2 = r + 2;
        auto n2 = oddr.query_leq(m + 1, r, c2);
        ll less2 = n2.cnt, sum2 = n2.sm;
        ll right = -get((m + 1) + 1, (r) + 1) + sum2 + (ll)(r - m - less2) * c2;
        return left + right;
    }

    ll compute_even(int l, int r) {
        int m = (l + r) >> 1;
        int c1 = -(l + 1);
        auto n1  = evenl.query_leq(l, m, c1);
        ll less1 = n1.cnt, sum1 = n1.sm;
        ll left  = get(l + 1, m + 1) + sum1 + (ll)(m - l + 1 - less1) * c1;

        int c2 = r + 2;
        auto n2  = evenr.query_leq(m + 1, r, c2);
        ll less2 = n2.cnt, sum2 = n2.sm;
        ll right = -get((m + 1) + 1, (r) + 1) + sum2 + (ll)(r - m - less2) * c2;
        return left + right;
    }
};

struct PASCAL {
    vvll prefix;
    vpii coord;
    vvi pascal;
    int limit;
    PASCAL(int _N) : limit(_N), coord(_N) {
        int cnt = 1;
        for(int r = 0; cnt < limit; r++) {
            pascal.pb(vi(r + 1));
            auto& curr = pascal.back();
            for(int i = 0; i < (int)curr.size(); i++) {
                curr[i] = cnt++;
                coord[curr[i]] = MP(r, i);
                if(cnt == limit) break;
            }
            const int N = curr.size();
            prefix.pb(vll(N + 1));
            auto& p = prefix.back();
            for(int i = 1; i <= N; i++) {
                p[i] = p[i - 1] + (ll)curr[i - 1] * curr[i - 1];
            }
        }
    }
};

