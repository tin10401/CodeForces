template<typename T>
class Treap {
private:
    struct TreapNode {
        int pri, size, f;
        T key;
        TreapNode* left;
        TreapNode* right;
        
        TreapNode(T key) : f(0), key(key), pri(rand()), size(1), left(nullptr), right(nullptr) {}
    };

    TreapNode* root;

    int size(TreapNode* treap) {
        if (!treap) return 0;
        return treap->size;
    }
    
    void unite(TreapNode* treap) {  
        treap->size = size(treap->left) + size(treap->right) + 1;
    }

    T flip(T bit) { 
        return !bit;
    }

    void apply(TreapNode* treap) {  
        if(!treap) return;
        if(treap->f) {  
            swap(treap->left, treap->right);
            if(treap->left) treap->left->f ^= 1;    
            if(treap->right) treap->right->f ^= 1;
            treap->f = 0;
        }
    }

    void split(TreapNode* treap, TreapNode*& left, TreapNode*& right, int k) {
        if (!treap) {
            left = right = nullptr;
            return;
        }
        apply(treap);
        if (size(treap->left) >= k) { // treap->key > k
            split(treap->left, left, treap->left, k);
            right = treap;
        } else {
            split(treap->right, treap->right, right, k - size(treap->left) - 1); // careful when split by value
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
            apply(left);
            merge(left->right, left->right, right);
            treap = left;
        } else {
            apply(right);
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
    Treap() : root(nullptr) {}
    
    ~Treap() {
        destroy(root);
    }

    void insert(T key) { 
        merge(root, root, new TreapNode(key));
    }
    
	void split_and_reverse(int l, int r) { // off_set by 1
        TreapNode* A, *B;
        split(root, root, A, l - 1); 
        split(A, A, B, r - l + 1);
        if(A) { 
            A->f ^= 1;  
            apply(A);
        }
        merge(root, root, A);   
        merge(root, root, B);
    }
	
	void insert_at(int k, T x) { // one base index
        if(size(root) < k) {
            cout << "Can't insert" << endl;
            return;
        }
        TreapNode* A;
        split(root, root, A, k - 1);
        merge(root, root, new TreapNode(x));
        merge(root, root, A);
    }

    void erase_at(int k) { // one base index
        if(size(root) < k) {
            cout << "Can't erase" << endl;
            return;
        }
        TreapNode*A, *B;
        split(root, root, A, k - 1);
        split(A, A, B, 1);
        merge(root, root, B);
    }
	
	void split_and_swap(int k) { // off_set by 1
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

    T get(int k) {  
        TreapNode* A, *B, *C;   
        split(root, A, B, k - 1);
        split(B, B, C, 1);  
        T ans = B->key;
        merge(root, A, B);  
        merge(root, root, C);
        return ans;
    }
	
	TreapNode* erase(int l, int r) {
        TreapNode* A, *B;
        split(root, root, A, l - 1);
        split(A, A, B, r);
        merge(root, root, B);
        return A;
    }

	void split2(TreapNode* treap, TreapNode*& left, TreapNode*& right, int k) {
        if (!treap) {
            left = right = nullptr;
            return;
        }
        apply(treap);
        if (treap->key > k) { // treap->key > k
            split2(treap->left, left, treap->left, k);
            right = treap;
        } else {
            split2(treap->right, treap->right, right, k); // careful when split by value
            left = treap;
        }
        unite(treap);
    }

    TreapNode* merge_treap(TreapNode* A, TreapNode* B) {
        if(!A) return B;
        if(!B) return A;
        if(A->pri < B->pri) swap(A, B);
        TreapNode*L, *R;
        split2(B, L, R, A->key);
        A->left = merge_treap(L, A->left);
        A->right = merge_treap(A->right, R);
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
        apply(treap);
        if(!treap) return;
        print(treap->left); 
        cout << treap->key;
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
 
template<class T, typename F = function<T(const T&, const T&)>>
class FW {  
    public: 
    int n, N;
    vt<T> root;    
    T DEFAULT;
    F func;
    FW(int n, T DEFAULT, F func) : func(func) { 
        this->n = n;    
        this->DEFAULT = DEFAULT;
        N = log2(n);
        root.rsz(n, DEFAULT);
    }
    
    void update(int id, T val) {  
        while(id < n) {    
            root[id] = func(root[id], val);
            id |= (id + 1);
        }
    }
    
    T get(int id) {   
        T res = DEFAULT;
        while(id >= 0) { 
            res = func(res, root[id]);
            id = (id & (id + 1)) - 1;
        }
        return res;
    }

    T queries_range(int left, int right) {  
        return get(right) - get(left - 1);
    }

    T queries_at(int i) {
        return queries_range(i, i);
    }
	
	void reset() {
		root.assign(n, 0);
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

template<class T>
class FW_2D {
    public:
    int n;
    vt<vt<T>> coord, root;
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
 
    void add_coord(int i, T x, bool is_up = true) {
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
        if(low > high) return 0;
        return point_query(i, high) - point_query(i, low - 1);
    }
 
    T range_queries(int l, int r, int low, int high) {
        if(l > r || low > high) return 0;
        return bit_range_queries(r, low, high) - bit_range_queries(l - 1, low, high);
    }

    T merge(T left, T right) {
    }
};

template<typename T, typename I = int, typename II = ll, typename F = function<T(const T, const T)>, typename G = function<void(int i, int left, int right, I)>>
class SGT { 
    public: 
    int n;  
    vt<T> root;
	vt<II> lazy;
    T DEFAULT;
    F func;
    G apply_func;
	SGT(int n, T DEFAULT, F func, G apply_func = [](int i, int left, int right, I val){}) : func(func), apply_func(apply_func) {    
        this->n = n;
        this->DEFAULT = DEFAULT;
		int k = 1;
        while(k < n) k <<= 1; 
        root.rsz(k << 1);    
        lazy.rsz(k << 1); // careful with initializing lazy_value
		// *** when doing merging close_interval, do middle, right instead of middle + 1, right for right child, and check for nullptr by right - left <= 1 instead of left == right like normal
		// and right <= start || left >= end instead of normally you don't have the '=' sign

    }
    
    void update_at(int id, T val) {  
        update_at(entireTree, id, val);
    }
    
    void update_at(iter, int id, T val) {  
		pushDown;
        if(left == right) { 
            root[i] = val;  
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update_at(lp, id, val);   
        else update_at(rp, id, val);   
        root[i] = func(root[lc], root[rc]);
    }

    void update_range(int start, int end, I val) { 
        update_range(entireTree, start, end, val);
    }
    
    void update_range(iter, int start, int end, I val) {    
        pushDown;   
        if(left > end || start > right) return; 
        if(left >= start && right <= end) { 
			apply_func(i, left, right, val);
            pushDown;   
            return;
        }
        int middle = midPoint;  
        update_range(lp, start, end, val);    
        update_range(rp, start, end, val);    
        root[i] = func(root[lc], root[rc]);
    }

    void push(iter) {   
        if(lazy[i] && left != right) {
			int middle = midPoint;
            apply_func(lp, lazy[i]), apply_func(rp, lazy[i]);
            lazy[i] = 0;
        }
    }

	T queries_at(int id) {
		return queries_at(entireTree, id);
	}
	
	T queries_at(iter, int id) {
		pushDown;
		if(left == right) {
			return root[i];
		}
		int middle = midPoint;
		if(id <= middle) return queries_at(lp, id);
		return queries_at(rp, id);
	}

    T queries_range(int start, int end) { 
        return queries_range(entireTree, start, end);
    }
    
    T queries_range(iter, int start, int end) {   
        pushDown;
        if(left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return root[i];   
        int middle = midPoint;  
        return func(queries_range(lp, start, end), queries_range(rp, start, end));
    }
	
	T get() {
		return root[0];
	}
	
	void print() {  
        print(entireTree);
        cout << endl;
    }
    
    void print(iter) {  
        pushDown;
        if(left == right) { 
            cout << root[i] << ' ';
            return;
        }
        int middle = midPoint;  
        print(lp);  print(rp);
    }

//    T merge(const T &left, const T &right) {
//        T res;
//        for (int a = 0; a < 2; ++a) {
//            for (int b = 0; b < (a ? 1 : 2); ++b) {
//                auto &curr = res.dp[a + b];
//                auto &L = left.dp[a];
//                auto &R = right.dp[b];
//                for(int i = 0; i < 2; i++) {
//                    for(int j = 0; j < 2; j++) {
//                        curr[i][j] = max({curr[i][j], 
//                            L[i][0] + R[0][j], 
//                            L[i][1] + R[0][j],
//                            L[i][0] + R[1][j]
//                        });
//                    }
//                }
//            }
//        }
//        return res;
//    }
};

template<class T, typename F = function<T(const T, const T)>>
class basic_segtree {
public:
    int n;    
    int size;  
    vt<T> root;
    F func;
    T DEFAULT;  
    
    basic_segtree(int n, T DEFAULT, F func) : n(n), DEFAULT(DEFAULT), func(func) {
        size = 1;
        while (size < n) size <<= 1;
        root.assign(size << 1, DEFAULT);
    }
    
    void update_at(int idx, T val) {
        idx += size, root[idx] = val;
        for (idx >>= 1; idx > 0; idx >>= 1) root[idx] = func(root[idx << 1], root[idx << 1 | 1]);
    }
    
    T queries_range(int l, int r) {
        T res_left = DEFAULT, res_right = DEFAULT;
        l += size, r += size;
        while (l <= r) {
            if ((l & 1) == 1) res_left = func(res_left, root[l++]);
            if ((r & 1) == 0) res_right = func(root[r--], res_right);
            l >>= 1; r >>= 1;
        }
        return func(res_left, res_right);
    }
	
	T queries_at(int idx) {
        return root[idx + size];
    }

    T get() {
        return root[1];
    }
};

template<class T, class I = int, typename F = function<T(const T, const T)>>
class iterative_lazy_segtree {
public:
    int n, size, h;
    vt<T> root;
    vt<I> lazy;
    F func;
    T DEFAULT;
    
    iterative_lazy_segtree(int n, T DEFAULT, F func) : n(n), DEFAULT(DEFAULT), func(func) {
        size = 1;
        while (size < n) size <<= 1;
        root.assign(size << 1, DEFAULT);
        lazy.assign(size << 1, 0);
        h = 0;
        for (int i = size; i > 0; i >>= 1) h++;
    }
    
    inline void update_at(int idx, T val) {
        if(idx < 0 || idx >= n) return;
        idx += size;
        for (int s = h; s > 0; s--) 
            push(idx >> s);
        root[idx] = val;
        for (idx /= 2; idx > 0; idx /= 2) {
            root[idx] = func(root[idx << 1], root[idx << 1 | 1]);
            root[idx] += lazy[idx];
        }
    }
    
    inline void update_range(int l, int r, I val) {
        if(l < 0 || r >= n || l > r) return;
        int L = l + size, R = r + size;
        int l0 = L, r0 = R;
        for (int s = h; s > 0; s--) {
            if ((L >> s) << s != L) push(L >> s);
            if ((R >> s) << s != R) push(R >> s);
        }
        while (L <= R) {
            if (L & 1) { apply(L, val); L++; }
            if (!(R & 1)) { apply(R, val); R--; }
            L >>= 1; R >>= 1;
        }
        rebuild(l0);
        rebuild(r0);
    }
    
    inline T queries_range(int l, int r) {
        if(l < 0 || r >= n || l > r) return DEFAULT;
        int L = l + size, R = r + size;
        for (int s = h; s > 0; s--) {
            if ((L >> s) << s != L) push(L >> s);
            if ((R >> s) << s != R) push(R >> s);
        }
        T res_left = DEFAULT, res_right = DEFAULT;
        while (L <= R) {
            if (L & 1) res_left = func(res_left, root[L++]);
            if (!(R & 1)) res_right = func(root[R--], res_right);
            L >>= 1; R >>= 1;
        }
        return func(res_left, res_right);
    }
    
    inline T queries_at(int idx) {
        if(idx < 0 || idx >= n) return DEFAULT;
        idx += size;
        for (int s = h; s > 0; s--)
            push(idx >> s);
        return root[idx];
    }
    
    inline T get() { return root[1]; }
    
private:
    inline void apply(int i, I val) {
        root[i] += val; 
        if (i < size)
            lazy[i] += val;
    }
    
    inline void push(int i) {
        if (lazy[i] != 0) {
            apply(i << 1, lazy[i]);
            apply(i << 1 | 1, lazy[i]);
            lazy[i] = 0;
        }
    }
    
    inline void rebuild(int i) {
        for (i /= 2; i > 0; i /= 2) {
            root[i] = func(root[i << 1], root[i << 1 | 1]);
            root[i] += lazy[i];
        }
    }
};

//struct info {
//    ll ans, suffix, prefix, sm;
//    info(ll x = -INF) : suffix(x), prefix(x), ans(max(0LL, x)), sm(x) {}
//};
//SGT<info> root(n, info(), [](const info& left, const info& right) {
//                                if(left.sm == -INF) return right;
//                                if(right.sm == -INF) return left;
//                                info res;  
//                                res.prefix = max(left.prefix, left.sm + right.prefix);
//                                res.suffix = max(right.suffix, right.sm + left.suffix);
//                                res.ans = max({left.ans, right.ans, left.suffix + right.prefix});
//                                res.sm = left.sm + right.sm;
//                                return res; 
//                            });

struct merge_sort_tree {
    int n;
    vvi arr;
    vi ans;
    int res = inf;
    merge_sort_tree(vi& a) : n(a.size()) {
        arr.rsz(n * 4);
        ans.rsz(n * 4);
        build(entireTree, a);
    }

    void build(iter, vi& a) {
        ans[i] = inf;
        for(int j = left; j <= right; j++) arr[i].pb(a[j]);
        srt(arr[i]);
        if(left == right) return;
        int middle = midPoint;
        build(lp, a);
        build(rp, a);
    }

    void update_range(int start, int end, int x) {
        update_range(entireTree, start, end, x);
    }

    void update_range(iter, int s, int e, int x) {
        if(left > e || s > right) return;
        if(s <= left && right <= e) {
            auto it = lb(all(arr[i]), x);
            int t = inf;
            if(it != end(arr[i])) t = min(t, abs(*it - x));
            if(it != begin(arr[i])) t = min(t, abs(*--it - x));
            ans[i] = min(ans[i], t);
            if(t >= res) return;
        }
        if(left == right) {
            res = min(res, ans[i]);
            return;
        }
        int middle = midPoint;
        update_range(rp, s, e, x);
        res = min(res, ans[rc]);
        update_range(lp, s, e, x);
        ans[i] = min(ans[lc], ans[rc]);
        res = min(res, ans[i]);
    }

    int queries_range(int left, int right) {
        return queries_range(entireTree, left, right);
    }

    int queries_range(iter, int s, int e) {
        if(left > e || s > right) return inf;
        if(s <= left && right <= e) return ans[i];
        int middle = midPoint;
        return min(queries_range(lp, s, e), queries_range(rp, s, e));
    }
};

// PERSISTENT SEGTREE
int t[MX * MK], ptr, root[MX * 120]; // log2 = MX * 200; careful to match root with the type of template below
pii child[MX * 120];
template<class T>
struct PSGT {
    int n;
    T DEFAULT;
    void assign(int n, T DEFAULT) {
        this->DEFAULT = DEFAULT;
        this->n = n;
    }

	void update(int &curr, int prev, int id, T delta, int left, int right) {  
        if(!curr) curr = ++ptr; // comment_out if regular PERSISTENT segtree
        root[curr] = root[prev];    
        child[curr] = child[prev];
        if(left == right) { 
			root[curr] = merge(root[curr], delta);
            return;
        }
        int middle = midPoint;
        if(id <= middle) update(child[curr].ff, child[prev].ff, id, delta, left, middle); // for regular_persistent segtree, do child[curr].ff or child[curr].ss = ++ptr;
        else update(child[curr].ss, child[prev].ss, id, delta, middle + 1, right);
        root[curr] = merge(root[child[curr].ff], root[child[curr].ss]);
    }

	T queries_at(int curr, int start, int end, int left, int right) { 
        if(!curr || left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return root[curr];
        int middle = midPoint;  
		return merge(queries_at(child[curr].ff, start, end, left, middle), queries_at(child[curr].ss, start, end, middle + 1, right));
    };

        
    T get(int curr, int prev, int k, int left, int right) {    
        if(root[curr] - root[prev] < k) return DEFAULT;
        if(left == right) return left;
        int leftCount = root[child[curr].ff] - root[child[prev].ff];
        int middle = midPoint;
        if(leftCount >= k) return get(child[curr].ff, child[prev].ff, k, left, middle);
        return get(child[curr].ss, child[prev].ss, k - leftCount, middle + 1, right);
    }

    T get(int l, int r, int k) {
        return get(t[r], t[l - 1], k, 0, n - 1);
    }

    void reset() {  
        for(int i = 0; i <= ptr; i++) { 
            root[i] = t[i] = 0;
            child[i] = {0, 0};
        }
        ptr = 0;
    }

    void add(int i, int& prev, int id, T delta) { 
        // t[i] = ++ptr // for regular PERSISTENT segtree
        update(t[i], prev, id, delta, 0, n - 1); // this is for 2d_segtree
        prev = t[i];
//        while(i < n) { 
//            update(t[i], t[i], id, delta, 0, n - 1);
//            i |= (i + 1);
//        }
    }

    T queries_at(int i, int start, int end) {
        return queries_at(t[i], start, end, 0, n - 1);
//        while(i >= 0) {
//            res += queries(t[i], start, end, 0, n - 1);
//            i = (i & (i + 1)) - 1;
//        }
    }

	T queries_range(int l, int r, int low, int high) {
        if(l > r || low > high) return DEFAULT;
        auto L = (l == 0 ? DEFAULT : queries_at(l - 1, low, high));
        auto R = queries_at(r, low, high);
        return R - L;
    }

    T merge(T left, T right) {
    }
};

template<class T>
struct SGT_2D {
    vt<vt<T>> root;
    int n, m, N;           
    T DEFAULT;             

    SGT_2D(int n, int m, T DEFAULT) {
        this->n = n;
        this->m = m;
        this->DEFAULT = DEFAULT;
        this->N = max(n, m); 
        root.resize(N * 2, vt<T>(N * 2)); // do 4 * N for recursive segtreee
    }

    void update_at(int x, int y, T value) {
        x += N; y += N;
        root[x][y] = value;
        for (int ty = y; ty > 1; ty >>= 1) {
            root[x][ty >> 1] = merge(root[x][ty], root[x][ty ^ 1]);
        }
        for (int tx = x; tx > 1; tx >>= 1) {
            for (int ty = y; ty >= 1; ty >>= 1) {
                root[tx >> 1][ty] = merge(root[tx][ty], root[tx ^ 1][ty]);
            }
        }
    }

    T queries_range(int start_x, int end_x, int start_y, int end_y) {
        start_x += N; end_x += N;    
        start_y += N; end_y += N;   
        T result = DEFAULT;

        while (start_x <= end_x) {
            if (start_x & 1) { 
                int sy = start_y, ey = end_y;
                while (sy <= ey) {
                    if (sy & 1) result = merge(result, root[start_x][sy++]);
                    if (!(ey & 1)) result = merge(result, root[start_x][ey--]);
                    sy >>= 1; ey >>= 1;
                }
                start_x++;
            }
            if (!(end_x & 1)) {
                int sy = start_y, ey = end_y;
                while (sy <= ey) {
                    if (sy & 1) result = merge(result, root[end_x][sy++]);
                    if (!(ey & 1)) result = merge(result, root[end_x][ey--]);
                    sy >>= 1; ey >>= 1;
                }
                end_x--;
            }
            start_x >>= 1;
            end_x >>= 1;
        }
        return result;
    }

    T merge(T A, T B) {
    }

//    void update_at(int x, int y, T v) {
//        update_at(x, y, v, 0, n - 1, 0, m - 1, 1, 1);
//    }
// 
//    void update_at(int x, int y, T val, int left_x, int right_x, int left_y, int right_y, int node_x, int node_y) {
//        if (left_x == right_x && left_y == right_y) {
//            root[node_x][node_y] = val;
//            return;
//        }
//        int mid_x = (left_x + right_x) / 2;
//        int mid_y = (left_y + right_y) / 2;
//        if (x <= mid_x && y <= mid_y) update_at(x, y, val, left_x, mid_x, left_y, mid_y, 2 * node_x, 2 * node_y); 
//        else if (x <= mid_x) update_at(x, y, val, left_x, mid_x, mid_y + 1, right_y, 2 * node_x, 2 * node_y + 1);
//        else if (y <= mid_y) update_at(x, y, val, mid_x + 1, right_x, left_y, mid_y, 2 * node_x + 1, 2 * node_y);
//        else update_at(x, y, val, mid_x + 1, right_x, mid_y + 1, right_y, 2 * node_x + 1, 2 * node_y + 1);
//        root[node_x][node_y] = merge(
//            root[2 * node_x][2 * node_y],
//            root[2 * node_x][2 * node_y + 1],
//            root[2 * node_x + 1][2 * node_y],
//            root[2 * node_x + 1][2 * node_y + 1]
//        );
//    }
// 
//    T queries_range(int start_x, int end_x, int start_y, int end_y) {
//        return queries_range(start_x, end_x, start_y, end_y, 0, n - 1, 0, m - 1, 1, 1);
//    }
// 
//    T queries_range(int start_x, int end_x, int start_y, int end_y, int left_x, int right_x, int left_y, int right_y, int node_x, int node_y) {
//        if (start_x > right_x || end_x < left_x || start_y > right_y || end_y < left_y) return DEFAULT;
//        if (start_x <= left_x && right_x <= end_x && start_y <= left_y && right_y <= end_y) return root[node_x][node_y];
//        int mid_x = (left_x + right_x) / 2;
//        int mid_y = (left_y + right_y) / 2;
//        return merge(
//            queries_range(start_x, end_x, start_y, end_y, left_x, mid_x, left_y, mid_y, 2 * node_x, 2 * node_y),
//            queries_range(start_x, end_x, start_y, end_y, left_x, mid_x, mid_y + 1, right_y, 2 * node_x, 2 * node_y + 1),
//            queries_range(start_x, end_x, start_y, end_y, mid_x + 1, right_x, left_y, mid_y, 2 * node_x + 1, 2 * node_y),
//            queries_range(start_x, end_x, start_y, end_y, mid_x + 1, right_x, mid_y + 1, right_y, 2 * node_x + 1, 2 * node_y + 1)
//        );
//    }
// 
//    T merge(T A, T B, T C, T D) {
//        return min(A, min(B, min(C, D)));
//    }
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
        umap<int, int> mp;  
        int N = pos.size();
        for(int i = 0; i < N; i++) mp[pos[i]] = i;
        for(auto& it : a) it = mp[it];

        vll dp(N);
        ll ans = 0;
        auto modify = [&](int x, int v) -> void {    
            if(pos[x] == 0) return;
            if(dp[x] == pos[x]) ans--;  
            dp[x] += v; 
            if(dp[x] == pos[x]) ans++;
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

template<typename T, typename F = function<T(const T&, const T&)>> // SparseTable<int, function<int(int, int)>>(vector, [](int x, int y) {return max(a, b);});
class SparseTable {
public:
    int n;
    vt<vt<T>> dp;
    vi log_table;
    F func;

    SparseTable(const vi& a, F func) : n(a.size()), func(func) {
        dp.rsz(n, vt<T>(floor(log2(n)) + 2));
        log_table.rsz(n + 1);
        for (int i = 2; i <= n; i++) log_table[i] = log_table[i / 2] + 1;
        for (int i = 0; i < n; i++) dp[i][0] = a[i];
        for (int j = 1; (1 << j) <= n; j++) {
            for (int i = 0; i + (1 << j) <= n; i++) {
                dp[i][j] = func(dp[i][j - 1], dp[i + (1 << (j - 1))][j - 1]);
            }
        }
    }

    T query(int L, int R) {
        assert(L >= 0 && R >= 0 && L < n && R < n && L <= R);
        int j = log_table[R - L + 1];
        return func(dp[L][j], dp[R - (1 << j) + 1][j]);
    }
};

template<typename T, typename F = function<T(const T&, const T&)>>
struct SparseTable2D {
    int n, m, LOGN, LOGM;
    vt<vt<vt<vt<T>>>> st;
    vt<int> logn, logm;
    F f;
    T DEFAULT;
    SparseTable2D(const vt<vt<T>> &a, T DEFAULT, F func) : f(func), DEFAULT(DEFAULT) {
        n = a.size();
        m = a[0].size();
        LOGN = log2(n) + 1;
        LOGM = log2(m) + 1;
        logn.rsz(n + 1, 0);
        logm.rsz(m + 1, 0);
        for (int i = 2; i <= n; i++) logn[i] = logn[i / 2] + 1;
        for (int j = 2; j <= m; j++) logm[j] = logm[j / 2] + 1;
        st.assign(LOGN, vt<vt<vt<T>>>(LOGM, vt<vt<T>>(n, vt<T>(m))));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                st[0][0][i][j] = a[i][j];
        for (int l = 1; l < LOGM; l++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j + (1 << l) <= m; j++) {
                    st[0][l][i][j] = f(st[0][l-1][i][j], st[0][l-1][i][j + (1 << (l-1))]);
                }
            }
        }
        for (int k = 1; k < LOGN; k++) {
            for (int l = 0; l < LOGM; l++) {
                for (int i = 0; i + (1 << k) <= n; i++) {
                    for (int j = 0; j < m; j++) {
                        st[k][l][i][j] = f(st[k-1][l][i][j], st[k-1][l][i + (1 << (k-1))][j]);
                    }
                }
            }
        }
    }
    T query(int r1, int c1, int r2, int c2) {
        if(r2 < r1 || c2 < c1 || r1 <= 0 || r2 > n || c1 <= 0 || c2 > m) return DEFAULT;
        int h = r2 - r1 + 1, w = c2 - c1 + 1;
        int k = logn[h], l = logm[w];
        T a1 = st[k][l][r1][c1],
          a2 = st[k][l][r2 - (1 << k) + 1][c1],
          a3 = st[k][l][r1][c2 - (1 << l) + 1],
          a4 = st[k][l][r2 - (1 << k) + 1][c2 - (1 << l) + 1];
        return f(f(a1, a2), f(a3, a4));
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

int root[MX * 120], lazy[MX * 120], ptr; // MX should be 1e5
pii child[MX * 120];
class implit_segtree {
    public:
    int n;
    implit_segtree(int n) {
        this->n = n;
        root[0] = a.queries(0, n - 1); // initialize
        lazy[0] = -1;
    } 

    void create_node(int& node, int left, int right) {
        if(node) return;
        node = ++ptr;
        lazy[node] = -1;
        root[node] = a.queries(left, right);
    }

    void update(int start, int end, int x) {
        update(entireTree, start, end, x);
    }

    void update(iter, int start, int end, int x) {
        pushDown;
        if(left > end || start > right) return;
        if(start <= left && right <= end) {
			apply(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        create_node(child[i].ff, left, middle);
        create_node(child[i].ss, middle + 1, right);
        update(child[i].ff, left, middle, start, end, x);
        update(child[i].ss, middle + 1, right, start, end, x);
        root[i] = merge(root[child[i].ff], root[child[i].ss]);
    }

	void apply(iter, int val) {
        root[i] = val * (right - left + 1);
        lazy[i] = val;
    }

    void push(iter) {
        if(lazy[i] != -1 && left != right) {
            int middle = midPoint;
            create_node(child[i].ff, left, middle);
            create_node(child[i].ss, middle + 1, right);
			apply(child[i].ff, left, middle, lazy[i]);
            apply(child[i].ss, middle + 1, right, lazy[i]);
            lazy[i] = -1;
        }
    }

    int merge(int left, int right) {
        return min(left, right);
    }

    int queries(int start, int end) {
        return queries(entireTree, start, end);
    }

    int queries(iter, int start, int end) {
        pushDown;
        if(start > right || left > end) return inf;
        if(start <= left && right <= end) return root[i];
        int middle = midPoint;
        create_node(child[i].ff, left, middle);
        create_node(child[i].ss, middle + 1, right);
        return merge(queries(child[i].ff, left, middle, start, end), queries(child[i].ss, middle + 1, right, start, end));
    }
	
	void update(int& i, int x) {
        update(i, 0, n - 1, x);
    }

    void update(int& i, int left, int right, int x) {
        if(!i) i = ++ptr;
        if(left == right) return;
        int middle = midPoint;
        if(x <= middle) update(child[i].ff, left, middle, x);
        else update(child[i].ss, middle + 1, right, x);
    }


    int merge_two_tree(int i, int j) {
        if(!i || !j) return i + j;
        child[i].ff = merge_two_tree(child[i].ff, child[j].ff);
        child[i].ss = merge_two_tree(child[i].ss, child[j].ss);
        return i;
    }

    void modify_two_tree(int& i, int& j, int start, int end) {
        modify(i, j, 0, n - 1, start, end);
    }

    void modify(int& i, int& j, int left, int right, int start, int end) {
        if(!i || left > end || start > right) return;
        if(!j) j = ++ptr;
        if(left >= start && right <= end) {
            j = merge_two_tree(i, j);
            i = 0;
            return;
        }
        int middle = midPoint;
        modify(child[i].ff, child[j].ff, left, middle, start, end);
        modify(child[i].ss, child[j].ss, middle + 1, right, start, end);
    }

};

template<class T>   
class SegTree_Graph { 
    public: 
    int n;  
    vt<vt<T>> graph;
    Undo_DSU root;
    vll ans;
    //vi pos;
    //int off;
	SegTree_Graph(int n, int q) : root(q) {    
        this->n = n;
        graph.rsz(n * 4); // n * 8 + 23;
        ans.rsz(n);
        //off = n * 4;
//        pos.rsz(n);
//        build(entireTree);
    }
    
    void update_range(int start, int end, T v) { 
        update_range(entireTree, start, end, v);
    }
    
    void update_range(iter, int start, int end, T v) {    
        if(left > end || start > right) return; 
        if(left >= start && right <= end) { 
            graph[i].pb(v);
            return;
        }
        int middle = midPoint;  
        update_range(lp, start, end, v);    
        update_range(rp, start, end, v);    
    }
 
    void run() {
        dfs(entireTree);
    }
 
    void dfs(iter) {
        int c = 0;
        for(auto& [u, v] : graph[i]) {
            if(root.merge(u, v, true)) c++;
        }
        if(left == right) {
            ans[left] = root.res;
        }
        else {
            int middle = midPoint;
            dfs(lp), dfs(rp);
        }
        while(c--) {
            root.rollBack();
        }
    }

//    void build(iter) { 
//        int u = i + off; // u here means from the parent can go to the children using off_set edges
//        if(left == right) {
//            graph[i].pb({u, 0});
//            graph[u].pb({i, 0});
//            pos[left] = i;
//            return;
//        }
//        int middle = midPoint;  
//        build(lp), build(rp); 
//        graph[lc].pb({i, 0});
//        graph[rc].pb({i, 0});
//        graph[u].pb({lc + off, 0});
//        graph[u].pb({rc + off, 0});
//    }
//
//    void add_edge(int u, int v, int w) {
//        u = pos[u], v = pos[v];
//        graph[u].pb({v, w});
//    }
//
//    void update(int start, int end, int u, int w, int type) { 
//        update(entireTree, start, end, pos[u], w, type);
//    }
//    
//    void update(iter, int start, int end, int u, int w, int type) {    
//        if(left > end || start > right) return; 
//        if(left >= start && right <= end) { 
//            if(type == 2) graph[u].pb({i + off, w});
//            else graph[i].pb({u, w});
//            return;
//        }
//        int middle = midPoint;  
//        update(lp, start, end, u, w, type);    
//        update(rp, start, end, u, w, type);    
//    }
};

template<class T>
class HLD {
    public:
    SGT<T> seg;
    vi id, tp, sz, parent;
    vt<T> a;
    int ct;
    vvi graph;
    int n;
    GRAPH g;
    HLD(vvi& graph, vt<T> a) : seg(graph.size(), 0), g(graph), graph(graph), n(graph.size()), a(a) {
        this->parent = g.parent;
        this->sz = g.subtree;
        ct = 0;
        id.rsz(n), tp.rsz(n), sz.rsz(n);
        dfs();
        for(int i = 0; i < n; i++) seg.update_at(id[i], a[i]);
    }
        
    void dfs(int node = 0, int par = -1, int top = 0) {   
        id[node] = ct++;    
        tp[node] = top;
        int nxt = -1, max_size = -1;    
        for(auto& nei : graph[node]) {   
            if(nei == par) continue;    
            if(sz[nei] > max_size) {   
                max_size = sz[nei]; 
                nxt = nei;  
            }   
        }   
        if(nxt == -1) return;   
        dfs(nxt, node, top);   
        for(auto& nei : graph[node]) {   
            if(nei != par && nei != nxt) dfs(nei, node, nei);  
        }   
    }

    void update(int i, T v) {
        a[i] = v;
        seg.update_at(id[i], v);
    }

    T queries(int node, int par) { // only query up to parent, don't include parent info
        T res = 0;    
        while(node != par && node != -1) {   
            if(node == tp[node]) {   
                res += a[node];
                node = parent[node];
            } else if(g.depth[tp[node]] > g.depth[par]) {   
                res += seg.queries_range(id[tp[node]], id[node]);    
                node = parent[tp[node]];
            } else {   
                res += seg.queries_range(id[par] + 1, id[node]);  
                break;  
            } 
        }   
        return res; 
    }

    T path_queries(int u, int v) { // remember to add parent info if needed
        int c = get_lca(u, v);
        T res = queries(u, c) * 2 - a[u] + a[c];
        res += queries(v, c) * 2 - a[v] + a[c];
        return res;
    }

    int get_dist(int a, int b) {
        return g.dist(a, b);
    }

    int get_lca(int a, int b) {
        return g.lca(a, b);
    }

    bool contain_all_node(int u, int v) {
        return path_queries(u, v) == get_dist(u, v);
    }
};

template<class T>
class SGT_BEAT {
    public:
    struct Node {
        T mx1, mx2, mn1, mn2, mx_cnt, mn_cnt, sm, ladd, lval;
        Node(T x = INF) : mx1(x), mx2(-INF), mn1(x), mn2(INF), mx_cnt(1), mn_cnt(1), sm(x), lval(INF), ladd(0) {}
    };
    int n;
    vt<Node> root;
    SGT_BEAT(int n) {
        this->n = n;
        root.rsz(n * 4);
    }

    void update_at(int id, T x) {
        update_at(entireTree, id, x);
    }

    void update_at(iter, int id, T x) {
        pushDown;
        if(left == right) {
            root[i] = Node(x);
            return;
        }
        int middle = midPoint;
        if(id <= middle) update_at(lp, id, x);
        else update_at(rp, id, x);
        root[i] = merge(root[lc], root[rc]);
    }

    Node merge(const Node left, const Node right) {
        if(left.mx1 == INF) return right;
        if(right.mx1 == INF) return left;
        Node res;
        res.sm = left.sm + right.sm;
        if(left.mx1 > right.mx1) {
            res.mx1 = left.mx1;
            res.mx_cnt = left.mx_cnt;
            res.mx2 = max(left.mx2, right.mx1);
        } else if(left.mx1 < right.mx1) {
            res.mx1 = right.mx1;
            res.mx_cnt = right.mx_cnt;
            res.mx2 = max(left.mx1, right.mx2);
        } else {
            res.mx1 = left.mx1;
            res.mx_cnt = left.mx_cnt + right.mx_cnt;
            res.mx2 = max(left.mx2, right.mx2);
        }

        if(left.mn1 < right.mn1) {
            res.mn1 = left.mn1;
            res.mn_cnt = left.mn_cnt;
            res.mn2 = min(left.mn2, right.mn1);
        } else if(left.mn1 > right.mn1) {
            res.mn1 = right.mn1;
            res.mn_cnt = right.mn_cnt;
            res.mn2 = min(right.mn2, left.mn1);
        } else {
            res.mn1 = left.mn1;
            res.mn_cnt = left.mn_cnt + right.mn_cnt;
            res.mn2 = min(left.mn2, right.mn2);
        }
        return res;
    }

    void update_min(int start, int end, T x) {
        update_min(entireTree, start, end, x);
    }

    void update_min(iter, int start, int end, T x) {
        pushDown;
        if(start > right || left > end || root[i].mx1 <= x) return;
        if(start <= left && right <= end && root[i].mx2 < x) {
            update_node_max(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_min(lp, start, end, x);
        update_min(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
    }

    void update_node_max(iter, T x) {
        root[i].sm += (x - root[i].mx1) * root[i].mx_cnt;
        if(root[i].mx1 == root[i].mn1) {
            root[i].mx1 = root[i].mn1 = x;
        } else if(root[i].mx1 == root[i].mn2) {
            root[i].mx1 = root[i].mn2 = x;
        } else {
            root[i].mx1 = x;
        }
    }

    void update_max(int start, int end, T x) {
        update_max(entireTree, start, end, x);
    }

    void update_max(iter, int start, int end, T x) {
        pushDown;
        if(left > end || start > right || x <= root[i].mn1) return;
        if(start <= left && right <= end && x < root[i].mn2) {
			update_node_min(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_max(lp, start, end, x);
        update_max(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
        
    }

    void update_node_min(iter, T x) {
        root[i].sm += (x - root[i].mn1) * root[i].mn_cnt;  
        if(root[i].mn1 == root[i].mx1) {
            root[i].mn1 = root[i].mx1 = x;
        } else if(root[i].mn1 == root[i].mx2) {
            root[i].mn1 = root[i].mx2 = x;
        } else {
            root[i].mn1 = x;
        }
    }

    void update_val(int start, int end, T x) {
        update_val(entireTree, start, end, x);
    }

    void update_val(iter, int start, int end, T x) {
        pushDown;
        if(start > right || left > end) return;
        if(start <= left && right <= end) {
            update_all(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_val(lp, start, end, x);
        update_val(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
    }

    void update_all(iter, T x) {
        root[i] = Node(x);
        T len = right - left + 1;
        root[i].sm = len * x;
        root[i].mx_cnt = root[i].mn_cnt = len;
        root[i].lval = x;

    }

    void update_add(int start, int end, T x) {
        update_add(entireTree, start, end, x);
    }

    void update_add(iter, int start, int end, T x) {
        pushDown;
        if(start > right || left > end) return;
        if(start <= left && right <= end) {
            add_val(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_add(lp, start, end, x);
        update_add(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
    }

    void add_val(iter, T x) {
        root[i].mx1 += x;
        if(root[i].mx2 != -INF) root[i].mx2 += x;
        root[i].mn1 += x;
        if(root[i].mn2 != INF) root[i].mn2 += x;
        root[i].sm += x * (right - left + 1);
        if(root[i].lval != INF) root[i].lval += x;
        else root[i].ladd += x;
    }

    void push(iter) {
        if(left == right) return;    
        int middle = midPoint;
        if(root[i].lval != INF) {
            update_all(lp, root[i].lval);
            update_all(rp, root[i].lval);
            root[i].lval = INF;
            return;
        }
        if(root[i].ladd) {
            add_val(lp, root[i].ladd);
            add_val(rp, root[i].ladd);
            root[i].ladd = 0;
        }
        if(root[i].mx1 < root[lc].mx1) update_node_max(lp, root[i].mx1);
        if(root[i].mn1 > root[lc].mn1) update_node_min(lp, root[i].mn1);
        if(root[i].mx1 < root[rc].mx1) update_node_max(rp, root[i].mx1);
        if(root[i].mn1 > root[rc].mn1) update_node_min(rp, root[i].mn1);
    }

    Node queries_range(int start, int end) {
        return queries_range(entireTree, start, end);
    }

    Node queries_range(iter, int start, int end) {
        pushDown;
        if(left > end || start > right) return Node();
        if(start <= left && right <= end) return root[i];
        int middle = midPoint;
        return merge(queries_range(lp, start, end), queries_range(rp, start, end));
    }
	
	Node queries_at(int id) {
		return queries_at(entireTree, id);
	}
	
	Node queries_at(iter, int id) {
		pushDown;
		if(left == right) {
			return root[i];
		}
		int middle = midPoint;
		if(id <= middle) return queries_at(lp, id);
		return queries_at(rp, id);
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

struct Node {
    int ch[2] = {0, 0};
    int fa = 0;       
    int size = 0;     
};

struct LCT {
    int n;             
    vector<Node> tree;  

    LCT(int n_) : n(n_) {
        tree.resize(n + 5);
        for (int i = 1; i < (int)tree.size(); ++i) // 1 base_index
            tree[i].size = 1;
    }

    inline void update(int x) {
        if (x == 0)
            return;
        tree[x].size = tree[ tree[x].ch[0] ].size + tree[ tree[x].ch[1] ].size + 1;
    }

    inline bool be_root(int x) {
        int f = tree[x].fa;
        return (tree[f].ch[0] != x && tree[f].ch[1] != x);
    }

    inline void rotate(int x) {
        int y = tree[x].fa;
        int flag = (tree[y].ch[0] == x);
        int tmp = tree[x].ch[flag];     
        if (!be_root(y))
            tree[ tree[y].fa ].ch[ tree[ tree[y].fa ].ch[1] == y ] = x;
        tree[x].fa = tree[y].fa;       
        tree[y].fa = x;
        tree[tmp].fa = tree[x].ch[flag] = y;
        tree[y].ch[flag ^ 1] = tmp;
        update(y);
    }

    inline void splay(int x) {
        while (!be_root(x)) {
            int y = tree[x].fa, z = tree[y].fa;
            if (be_root(y))
                rotate(x);
            else {
                if ((tree[z].ch[0] == y) ^ (tree[y].ch[0] == x)) rotate(x);
                else rotate(y);
                rotate(x);
            }
        }
        update(x);
    }

    inline void access(int x) {
        for (int y = 0; x; x = tree[x].fa) {
            splay(x);
            tree[x].ch[1] = y;
            update(x);
            y = x;
        }
    }

    inline pii query(int x) {
        access(x);
        splay(n + 1);
        int tmp = 0;
        for (tmp = tree[n + 1].ch[1]; tree[tmp].ch[0]; tmp = tree[tmp].ch[0]) {}
        return {tmp, tree[n + 1].size - 1};
    }

    inline void cut(int x, int y) {
        access(x);
        splay(y);
        tree[y].ch[1] = 0;
        tree[x].fa = 0;
        update(y);
    }

    inline void link(int x, int y) {
        tree[x].fa = y;
    }
};

class BITSET {
public:
    using ubig = unsigned long long;
    int sz;
    vector<ubig> blocks;
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
