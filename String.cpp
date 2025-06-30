class Binary_Trie { 
    struct Node {
        int c[2];
        int cnt;
        Node() {
            mset(c, 0);
            cnt = 0;
        }
    };
    public:
    static vt<Node> T; // careful with static if no merging needed
    int m;
    int root;
    Binary_Trie(int m) : m(m) { // max number of bit
        root = new_node();
    }

    int new_node() {
        T.pb(Node());
        return T.size() - 1;
    }
    
    void insert(ll num, int v = 1) {  
        int curr = root;   
        for(int i = m - 1; i >= 0; i--) {  
            int bits = (num >> i) & 1;  
            if(!T[curr].c[bits]) {
                T[curr].c[bits] = new_node();
            }
            curr = T[curr].c[bits];
            T[curr].cnt += v;
        }
		// dfs_insert(root, num, m - 1);
    }
	
	void dfs_insert(int curr, ll num, int bit) {
		if(bit == -1) {
            T[curr].cnt = 1;
			return;
		}
        int b = (num >> bit) & 1;
        if(!T[curr].c[b]) {
            T[curr].c[b] = new_node();
        }
        int nxt = T[curr].c[b];
        dfs_insert(nxt, num, bit - 1);
        T[curr].cnt = T[nxt].cnt + (T[curr].c[!b] ? T[T[curr].c[!b]].cnt : 0);
    }

    void merge(Binary_Trie& other) {
        root = merge_root(root, other.root);
        other.root = 0;
    }

    int merge_root(int u, int v) {
        if(u == 0) return v; 
        if(v == 0) return u;
        T[u].cnt += T[v].cnt;
        for(int bit = 0; bit < 2; bit++) {
            int cu = T[u].c[bit];
            int cv = T[v].c[bit];
            if(!cu) {
                T[u].c[bit] = cv;
            } else {
                T[u].c[bit] = merge_root(cu, cv);
            }
        }
        return u;
    }
        
    ll max_xor(ll num) {  
        ll res = 0, curr = root;
        for(int i = m - 1; i >= 0; i--) {  
            int bits = (num >> i) & 1;  
            int other = T[curr].c[!bits];
            if(other && T[other].cnt) {
                curr = T[curr].c[!bits];
                res |= (1LL << i);
            }
            else {  
                curr = T[curr].c[bits];
            }
            if(!curr) break;
        }
        return res;
    }
        
    ll min_xor(ll num) {  
        ll res = num, curr = root;
        for(int i = m - 1; i >= 0; i--) {  
            int bits = (num >> i) & 1;  
            int same = T[curr].c[bits];
            if(same && T[same].cnt) {
                curr = T[curr].c[bits];
                if(bits) res ^= (1LL << i);
            }
            else {  
                curr = T[curr].c[!bits];
                if(!bits) res ^= (1LL << i);
            }
            if(!curr) break;
        }
        return res;
    }
	
	ll count_less_than(ll a, ll b) {
        int curr = root;
        ll res = 0;
        for(int i = m - 1; i >= 0; i--) {
            int bits = (a >> i) & 1;
            int b_bits = (b >> i) & 1;
            if(b_bits) {
				if(T[curr].c[bits]) {
					res += T[T[curr].c[bits]].cnt;
				}
                curr = T[curr].c[!bits];
            }
            else {
                curr = T[curr].c[bits];
            }
            if(!curr) break;
        }
        // res += T[curr].cnt; // remove if count equal to as well
        return res;
    }
	
	ll count_greater_than(ll a, ll b) {
        int curr = root;
        ll res = 0;
        for(int i = m - 1; i >= 0; i--) {
            int bits = (a >> i) & 1;
            int b_bits = (b >> i) & 1;
            if(b_bits == 0 && T[curr].c[!bits]) {
                res += T[T[curr].c[!bits]].cnt;
            }
            curr = T[curr].c[b_bits ^ bits];
            if(!curr) break;
        }
        // res += T[curr].cnt; // remove if count equal to as well
        return res;
    }

	
	ll find_mex(ll x) { // find a first missing number
        ll mex = 0, curr = root;
        for(int i = m - 1; i >= 0; i--) {
            int bit = (x >> i) & 1;
            int c = T[curr].c[bit] ? T[T[curr].c[bit]].cnt : 0;
            if(c < (1LL << i)) {
                curr = T[curr].c[bit];
            }
            else {
                mex |= (1LL << i);
                curr = T[curr].c[!bit];
            }
            if(!curr) break;
        }
        return mex;
    }

    void clear() {
        vt<Node>().swap(T);
        root = new_node();
    }

}; vt<Binary_Trie::Node> Binary_Trie::T;

struct PERSISTENT_TRIE {
    struct Node {
        int level, cnt, c[2]; 
        Node() : level(0), cnt(0) {
            mset(c, 0); 
        }
    };
    int new_node() {
        T.pb(Node());
        return T.size() - 1;
    }
    vt<Node> T;
    vi root;
    int m;
    PERSISTENT_TRIE(int n, int bit) : m(bit) {
        new_node();
        root.rsz(n);
    }

    void add(int rt, int prev, int num, int v = 1, int lev = 0) {
        root[rt] = insert(prev, num, v, lev);
    }
    
    int insert(int prev, int num, int v = 1, int lev = 0) {   
        int newRoot = new_node();
        int curr = newRoot;
        for(int i = m - 1; i >= 0; i--) {   
            int bits = (num >> i) & 1;  
            T[curr].c[!bits] = T[prev].c[!bits];
            T[curr].c[bits] = new_node();
            prev = T[prev].c[bits];   
            curr = T[curr].c[bits];   
            T[curr].level = lev;
            T[curr].cnt = T[prev].cnt + v;
        }
        return newRoot;
    }

    int max_xor(int curr, int num, int lev = 0) {
        int res = 0;
        for(int i = m - 1; i >= 0; i--) {
            int bits = (num >> i) & 1;
            int nxt = T[curr].c[!bits];
            if(nxt && T[nxt].cnt && T[nxt].level >= lev) {
                res |= 1LL << i;
                curr = nxt;
            } else {
                curr = T[curr].c[bits];
            }
            if(!curr) break;
        }
        return res;
    }

    int min_xor(int curr, int num, int lev = 0) {
        int res = num;
        for(int i = m - 1; i >= 0; i--) {
            int bits = (num >> i) & 1;
            int nxt = T[curr].c[bits];
            if(nxt && T[nxt].cnt && T[nxt].level >= lev) {
                curr = nxt;
                if(bits) res ^= 1LL << i;
            }
            else {
                curr = T[curr].c[!bits];
                if(!bits) res ^= 1LL << i;
            }
            if(!curr) break;
        }
        return res;
    }

    int find_kth(vpii curr, int x, int k) { // https://toph.co/p/jontrona-of-liakot
        for(auto& [l, r] : curr) {
            l = root[l];
            r = root[r];
        }
        int res = 0;
        for(int i = m - 1; i >= 0; i--) {
            int bits = (x >> i) & 1;
            int same_count = 0;
            for(auto& [l, r] : curr) {
                same_count += (T[T[r].c[bits]].cnt - T[T[l].c[bits]].cnt);
            }
            if(same_count >= k) {
                for(auto& [l, r] : curr) {
                    l = T[l].c[bits];
                    r = T[r].c[bits];
                }
                continue;
            }
            k -= same_count;
            for(auto& [l, r] : curr) {
                l = T[l].c[!bits];
                r = T[r].c[!bits];
            }
            res |= 1LL << i;
        } 
        return res;
    }
};

struct Trie {
    struct Node {
        int c[26];
        Node() {
            mset(c, 0);
        }
    };
    vt<Node> T;
    char off;
    Trie(char _off) : off(_off) {
        T.pb(Node());
    }

    int new_node() {
        T.pb(Node());
        return T.size() - 1;
    }

    int get(char c) {
        return c - off;
    }

    void insert(const string& S, int v = 1) {
        int curr = 0;
        for(auto& ch : S) {
            int j = get(ch);
            if(!T[curr].c[j]) {
                T[curr].c[j] = new_node();
            }
            curr = T[curr].c[j];
        }
    }
};

struct aho_corasick {
    static const int SIGMA = 26;
    struct Node {
        int c[SIGMA], link[SIGMA], sfx, dict, is_end, cnt;
        Node() {
            mset(c, 0);
            mset(link, 0);
            sfx = dict = is_end = 0;
            is_end = 0;
            cnt = 0;
        }
    };
    vt<Node> T;
 
    char off;
    aho_corasick(char _off) : off(_off) { init(); }
 
    int get(char ch) {
        return ch - off;
    }
 
    int new_node(int x = -1) {
        T.pb(Node());
        return T.size() - 1;
    }
 
    void init() {
        T.pb(Node());
    }
 
    void insert(const string &s) {
        int u = 0;
        for(char ch: s) {
            int x = get(ch);
            if(!T[u].c[x]) {
                T[u].c[x] = new_node();
            }
            u = T[u].c[x];
        }
        T[u].is_end = true;
        T[u].cnt++;
    }

	bool built = false;
    void build() {
		built = true;
        queue<int> q;
        q.push(0);
        while(!q.empty()) {
            int par = q.front(); q.pop();
            for(int x = 0; x < SIGMA; x++) {
                int child = T[par].c[x];
                if(child) {
					int suff = T[T[par].sfx].link[x];
                    T[par].link[x] = par;
                    if(par != 0 && T[suff].c[x]) T[child].sfx = T[suff].c[x];
                    else T[child].sfx = 0;
                    T[child].dict = T[T[child].sfx].is_end ? T[child].sfx : T[T[child].sfx].dict;
                    T[child].cnt += T[T[child].sfx].cnt;
                    q.push(child);
                } else {
                    T[par].link[x] = T[T[par].sfx].link[x];
                }
            }
        }
    }

    int process(int &prev, char ch) { // return the number of nodes ending if moving to ch from prev
        if(!built) {
            built = true;
            build();
        }
        prev = advance(prev, ch);
        int curr = prev;
        int now = 0;
        return T[curr].cnt;
    }

    int advance(int u, char ch) {
        if(!built) {
            built = true;
            build();
        }
        return T[T[u].link[get(ch)]].c[get(ch)];
    }
 
    vi query(const string& s) {
        if(!built) {
            built = true;
            build();
        }
        int prev = 0;
        int n = s.size();
        vi ans(n);
        for(int i = 0; i < n; i++) {
            ans[i] = process(prev, s[i]);
        }
        return ans;
    }
};

const int N = 1e5 + 9;
struct aho_corasick_static {
  int cnt[N], link[N], psz;
  map<char, int> to[N];

  void clear() {
    for(int i = 0; i < psz; i++)
      cnt[i] = 0, link[i] = -1, to[i].clear();

    psz = 1;
    link[0] = -1;
    cnt[0] = 0;
  }

  aho_corasick_static() {
    psz = N - 2;
    clear();
  }

  void add_word(string s) {
    int u = 0;
    for(char c : s) {
      if(!to[u].count(c)) to[u][c] = psz++;
      u = to[u][c];
    }

    cnt[u]++;
  }

  void push_links() {
    queue<int> Q;
    int u, v, j;
    char c;

    Q.push(0);
    link[0] = -1;

    while(!Q.empty()) {
      u = Q.front();
      Q.pop();

      for(auto it : to[u]) {
        v = it.second;
        c = it.first;
        j = link[u];

        while(j != -1 && !to[j].count(c)) j = link[j];
        if(j != -1) link[v] = to[j][c];
        else link[v] = 0;

        cnt[v] += cnt[link[v]];
        Q.push(v);
      }
    }
  }

  int get_count(string p) {
    int u = 0, ans = 0;
    for(char c : p) {
      while(u != -1 && !to[u].count(c)) u = link[u];
      if(u == -1) u = 0;
      else u = to[u][c];
      ans += cnt[u];
    }

    return ans;
  }
};

struct aho_corasick {
  vector<string> li[20];
  aho_corasick_static ac[20];

  void clear() {
    for(int i = 0; i < 20; i++) {
      li[i].clear();
      ac[i].clear();
    }
  }

  aho_corasick() {
    clear();
  }

  void add_word(string s) {
    int pos = 0;
    for(int l = 0; l < 20; l++)
      if(li[l].empty()) {
        pos = l;
        break;
      }

    li[pos].push_back(s);
    ac[pos].add_word(s);

    for(int bef = 0; bef < pos; bef++) {
      ac[bef].clear();
      for(string s2 : li[bef]) {
        li[pos].push_back(s2);
        ac[pos].add_word(s2);
      }

      li[bef].clear();
    }

    ac[pos].push_links();
  }
  //sum of occurrences of all patterns in this string
  int get_count(string s) {
    int ans = 0;
    for(int l = 0; l < 20; l++)
      ans += ac[l].get_count(s);

    return ans;
  }
};
string s[N];
aho_corasick aho;

struct KMP {
    int n;
    string t;
    vi prefix;
    vvi dp; // quick linking
    char c;
    KMP() {}
    KMP(const string& t, char c) : t(t), c(c) {
        n = t.size();
        dp.rsz(n, vi(26));
        prefix.rsz(n);
        build();
        // property of finding period by kmp : if(len % (len - kmp[i]) == 0) period = len - kmp[i], otherwise period = len
        // to check if substring s[l, r] is k period, we just check the if s[l + k, r] == s[l, r - k]
    }

    void build() {
        for(int i = 1, j = 0; i < n; i++) { 
            while(j && t[i] != t[j]) j = prefix[j - 1]; 
            if(t[i] == t[j]) prefix[i] = ++j;
        }
        int n = t.size();
        for(int j = 0; j < 26; j++) {
            dp[0][j] = (t[0] == char(c + j) ? 1 : 0);
        }
        for(int i = 1; i < n; i++) {
            for(int j = 0; j < 26; j++) {
                if(t[i] == char(c + j)) {
                    dp[i][j] = i + 1;
                } else {
                    dp[i][j] = dp[prefix[i - 1]][j];
                }
            }
        }
    }

    int count_substring(const string& s) { // s is main string, t is pattern
        int N = s.size();
        int cnt = 0;
        //        vi occur;
        for(int i = 0, j = 0; i < N;) {
            if(s[i] == t[j]) i++, j++;
            else if(j) j = prefix[j - 1];
            else i++;
            if(j == n) {
                //                occur.pb(i - n);
                //                j = prefix[j - 1];
                cnt++;
                j = 0;
            }
        }
        return cnt;
    }

    ll count_substring(const vt<pair<char, int>>& a, const vt<pair<char, int>>& b) { // https://codeforces.com/contest/631/problem/D
        // compress form of [char, occurences] of s and t, count occurences of t in s
        vt<pair<char, ll>> s, t;
        for (auto &p : a) {
            if (!s.empty() && s.back().ff == p.ff) s.back().ss += p.ss;
            else s.emplace_back(p.ff, p.ss);
        }
        for (auto &p : b) {
            if (!t.empty() && t.back().ff == p.ff) t.back().ss += p.ss;
            else t.emplace_back(p.ff, p.ss);
        }
        int n = s.size(), m = t.size();
        if (n < m) return 0;
        ll ans = 0;
        if (m == 1) {
            for (int i = 0; i < n; i++)
                if (s[i].ff == t[0].ff && s[i].ss >= t[0].ss)
                    ans += s[i].ss - t[0].ss + 1;
            return ans;
        }
        if (m == 2) {
            for (int i = 0; i + 1 < n; i++)
                if (s[i].ff == t[0].ff && s[i].ss >= t[0].ss
                        && s[i + 1].ff == t[1].ff && s[i + 1].ss >= t[1].ss)
                    ans++;
            return ans;
        }
        int k = m - 2;
        vt<pair<char, ll>> mid;
        for (int i = 1; i <= k; i++) mid.pb(t[i]); // the len can be >= between first and last so we must match exactly the middle and deal with them later
        vi lps(k);
        for (int i = 1, len = 0; i < k; i++) {
            while (len && mid[i] != mid[len]) len = lps[len - 1];
            if (mid[i] == mid[len]) len++;
            lps[i] = len;
        }
        for (int i = 0, j = 0; i < n; i++) {
            while (j > 0 && (j >= k || s[i] != mid[j])) j = lps[j - 1];
            if (s[i] == mid[j]) j++;
            if (j == k) {
                int st = i - k, en = i + 1;
                if (st >= 0 && en < n
                        && s[st].ff == t[0].ff && s[st].ss >= t[0].ss
                        && s[en].ff == t[m - 1].ff && s[en].ss >= t[m - 1].ss)
                    ans++;
                j = lps[j - 1];
            }
        }
        return ans;
    }
};

struct Z {
    static vi get_z_vector(const string &s) {
        int n = s.size(), l = 0, r = 0;
        vi z(n);
        for(int i = 1; i < n; i++) {
            if(i <= r) {
                z[i] = min(r - i + 1, z[i - l]);
            }
            while(i + z[i] < n && s[z[i]] == s[i + z[i]]) {
                z[i]++;
            }
            if(i + z[i] - 1 > r) {
                l = i;
                r = i + z[i] - 1;
            }
        }
        return z;
    }

    static string concatnate(const string& s, const string& t) { // minimum len containing both s and t as substring
        if(s.find(t) != string::npos) return s;
        if(t.find(s) != string::npos) return t;
        int n = s.size(), m = t.size(), N = n + m;
        auto z = get_z_vector(t + s);
        for(int i = m; i < N; i++) {
            if(i + z[i] >= N) {
                return s + t.substr(N - i);
            }
        }
        return s + t;
    }

    static int overlap(const string &a, const string &b) {
        if(a.find(b) != string::npos) return b.size();
        if(b.find(a) != string::npos) return a.size();
        int na = a.size(), nb = b.size();
        int k = min(na, nb);
        string t = b + "#" + a.substr(na - k);
        vi z = get_z_vector(t);
        int best = 0;
        int L = b.size(), N = t.size();
        for(int i = L + 1; i < N; i++) {
            if(i + z[i] == N) {
                best = max(best, z[i]);
            }
        }
        return best;
    }

    static string super_str(const vs &S) { // return shortest string consist all of string in S as substring
                                           // only works for n < 20
        int N0 = S.size();
        vb remove(N0, false);
        for(int i = 0; i < N0; i++) {
            if(!remove[i]) {
                for(int j = 0; j < N0; j++) {
                    if(i != j && !remove[j]) {
                        if(S[i].find(S[j]) != string::npos) remove[j] = true;
                    }
                }
            }
        }
        vs A;
        for(int i = 0; i < N0; i++) if(!remove[i]) A.pb(S[i]);
        int N = A.size();
        if(N == 0) return "";

        vvi ov(N, vi(N));
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(i != j) ov[i][j] = overlap(A[i], A[j]);
            }
        }

        int ALL = 1 << N;
        vvi dp(ALL, vi(N)), par(ALL, vi(N));
        const int INF = 1e9;
        for(int m = 0; m < ALL; m++) {
            for(int i = 0; i < N; i++) {
                dp[m][i] = INF, par[m][i] = -1;
            }
        }

        for(int i = 0; i < N; i++) dp[1 << i][i] = A[i].size();

        for(int mask = 1; mask < ALL; mask++) {
            for(int last = 0; last < N; last++) {
                if(have_bit(mask, last)) {
                    int cur = dp[mask][last];
                    if(cur == INF) continue;
                    for(int nxt = 0; nxt < N; nxt++) {
                        if(!have_bit(mask, nxt)) {
                            int nm = mask | (1 << nxt);
                            int cand = cur + int(A[nxt].size()) - ov[last][nxt];
                            if(cand < dp[nm][nxt]) {
                                dp[nm][nxt] = cand;
                                par[nm][nxt] = last;
                            }
                        }
                    }
                }
            }
        }

        int full = ALL - 1, last = 0;
        for(int i = 1; i < N; i++)
            if(dp[full][i] < dp[full][last]) last = i;

        vi seq;
        for(int mask = full; mask;) {
            seq.pb(last);
            int p = par[mask][last];
            mask ^= 1 << last;
            last = p;
        }
        rev(seq);

        string res = A[seq[0]];
        for(int t = 1; t < (int)seq.size(); t++) {
            int i = seq[t - 1], j = seq[t];
            res += A[j].substr(ov[i][j]);
        }
        return res;
    }
};

https://toph.co/p/unique-substrings-query
const int HASH_COUNT = 2;
vll base, mod;
ll p[HASH_COUNT][MX];
void initGlobalHashParams() {
    if (!base.empty() && !mod.empty()) return;
	vll candidateBases = {
        10007ULL, 10009ULL, 10037ULL, 10039ULL, 10061ULL, 10067ULL, 10069ULL, 10079ULL, 10091ULL, 10093ULL,
        10099ULL, 10103ULL, 10111ULL, 10133ULL, 10139ULL, 10141ULL, 10151ULL, 10159ULL, 10163ULL, 10169ULL,
        10177ULL, 10181ULL, 10193ULL, 10211ULL, 10223ULL, 10243ULL, 10247ULL, 10253ULL, 10259ULL, 10267ULL,
        10271ULL, 10273ULL, 10289ULL, 10301ULL, 10303ULL, 10313ULL, 10321ULL, 10331ULL, 10333ULL, 10337ULL,
        10343ULL, 10357ULL, 10369ULL, 10391ULL, 10399ULL, 10427ULL, 10429ULL, 10433ULL, 10453ULL, 10457ULL,
        10459ULL, 10463ULL, 10477ULL, 10487ULL, 10499ULL, 10501ULL, 10513ULL, 10529ULL, 10531ULL, 10559ULL,
        10567ULL, 10589ULL, 10597ULL, 10601ULL, 10607ULL, 10613ULL, 10627ULL, 10631ULL, 10639ULL, 10651ULL,
        10657ULL, 10663ULL, 10667ULL, 10687ULL, 10691ULL, 10709ULL, 10711ULL, 10723ULL, 10729ULL, 10733ULL,
        10739ULL, 10753ULL, 10771ULL, 10781ULL, 10789ULL, 10799ULL, 10831ULL, 10837ULL, 10847ULL, 10853ULL,
        10859ULL, 10861ULL, 10867ULL, 10883ULL, 10889ULL, 10891ULL, 10903ULL, 10909ULL, 10937ULL, 10939ULL
    };

    vll candidateMods = {
        1000000007ULL, 1000000009ULL, 1000000021ULL, 1000000033ULL, 1000000087ULL, 1000000093ULL, 1000000097ULL, 1000000103ULL, 1000000123ULL, 1000000181ULL,
        1000000207ULL, 1000000223ULL, 1000000241ULL, 1000000271ULL, 1000000289ULL, 1000000297ULL, 1000000321ULL, 1000000349ULL, 1000000363ULL, 1000000403ULL,
        1000000409ULL, 1000000411ULL, 1000000427ULL, 1000000433ULL, 1000000439ULL, 1000000447ULL, 1000000453ULL, 1000000459ULL, 1000000483ULL, 1000000513ULL,
        1000000531ULL, 1000000579ULL, 1000000607ULL, 1000000613ULL, 1000000637ULL, 1000000663ULL, 1000000711ULL, 1000000753ULL, 1000000787ULL, 1000000801ULL,
        1000000829ULL, 1000000891ULL, 1000000901ULL, 1000000931ULL, 1000000951ULL, 1000000993ULL, 1000001011ULL, 1000001021ULL, 1000001053ULL, 1000001097ULL,
        1000001143ULL, 1000001161ULL, 1000001163ULL, 1000001179ULL, 1000001193ULL, 1000001231ULL, 1000001263ULL, 1000001303ULL, 1000001311ULL, 1000001351ULL,
        1000001369ULL, 1000001431ULL, 1000001453ULL, 1000001503ULL, 1000001531ULL, 1000001593ULL, 1000001617ULL, 1000001649ULL, 1000001663ULL, 1000001703ULL,
        1000001753ULL, 1000001783ULL, 1000001801ULL, 1000001853ULL, 1000001871ULL, 1000001957ULL, 1000001963ULL, 1000001969ULL, 1000002003ULL, 1000002029ULL,
        1000002043ULL, 1000002051ULL, 1000002061ULL, 1000002083ULL, 1000002101ULL, 1000002133ULL, 1000002161ULL, 1000002179ULL, 1000002253ULL, 1000002271ULL
    };
								 
	unsigned seed = chrono::steady_clock::now().time_since_epoch().count();
    shuffle(all(candidateBases), default_random_engine(seed));
    shuffle(all(candidateMods), default_random_engine(seed + 1));

    base.rsz(HASH_COUNT);
    mod.rsz(HASH_COUNT);
    for(int i = 0; i < HASH_COUNT; i++) {
        mod[i] = candidateMods[i];
        base[i] = candidateBases[i];
    }
    p[0][0] = p[1][0] = 1;
    for(int i = 1; i < MX; i++) {
        for(int j = 0; j < HASH_COUNT; j++) {
            p[j][i] = (p[j][i - 1] * base[j]) % mod[j];
        }
    }
}
static const bool _hashParamsInitialized = [](){
    initGlobalHashParams();
    return true;
}();

template<class T = string>
struct RabinKarp {
    // careful with the + 1 for the 0 hash
    vll prefix[HASH_COUNT], suffix[HASH_COUNT];
    int n;
    string s;

    RabinKarp() : n(0) {
        for(int i = 0; i < HASH_COUNT; i++) {
            prefix[i].pb(0);
            suffix[i].pb(0);
        }
    }

    RabinKarp(const T &s) : s(s) {
        n = s.size();
        for (int i = 0; i < HASH_COUNT; i++) {
            prefix[i].rsz(n + 1, 0);
            suffix[i].rsz(n + 1, 0);
        }
        for (int j = 1; j <= n; j++) {
            int x = s[j - 1] - 'a' + 1;
            int y = s[n - j] - 'a' + 1;
            for (int i = 0; i < HASH_COUNT; i++) {
                prefix[i][j] = (prefix[i][j - 1] * base[i] + x) % mod[i];
                suffix[i][j] = (suffix[i][j - 1] * base[i] + y) % mod[i];
            }
        }
    }

    void insert(int x) {
        for (int i = 0; i < HASH_COUNT; i++) {
            ll v = (prefix[i].back() * base[i] + x) % mod[i];
            prefix[i].pb(v);
        }
        n++;
    }

    void pop_back() {
        for (int i = 0; i < HASH_COUNT; i++) {
            prefix[i].pop_back();
        }
        n--;
    }

    int size() {
        return n;
    }
    
    ll get() {
        return get_hash(0, n);
    }

    ll get_hash(int l, int r) const {
        if (l < 0 || r > n || l > r) return 0;
        ll hash0 = prefix[0][r] - (prefix[0][l] * p[0][r - l] % mod[0]);
        hash0 = (hash0 % mod[0] + mod[0]) % mod[0];
        ll hash1 = prefix[1][r] - (prefix[1][l] * p[1][r - l] % mod[1]);
        hash1 = (hash1 % mod[1] + mod[1]) % mod[1];
        return (hash0 << 32) | hash1;
    }

    ll get_rev_hash(int l, int r) const {
        if(l < 0 || r > n || l >= r) return 0;
        ll h0 = suffix[0][r] - (suffix[0][l] * p[0][r - l] % mod[0]);
        ll h1 = suffix[1][r] - (suffix[1][l] * p[1][r - l] % mod[1]);
        if(h0 < 0) h0 += mod[0];
        if(h1 < 0) h1 += mod[1];
        return (h0 << 32) | h1;
    }

    bool is_palindrome(int l, int r) const {
        if(l > r) return true;
        return get_hash(l, r + 1) == get_rev_hash(n - 1 - r, n - l);
    }
    
    bool diff_by_one_char(RabinKarp &a, int offSet = 0) {
        int left = 0, right = n, rightMost = -1;
        while (left <= right) {
            int middle = left + (right - left) / 2;
            if (a.get_hash(offSet, middle + offSet) == get_hash(0, middle)) {
                rightMost = middle;
                left = middle + 1;
            } else {
                right = middle - 1;
            }
        }
        return a.get_hash(rightMost + 1 + offSet, offSet + n) == get_hash(rightMost + 1, n);
    }
    
    ll combine_hash(pll a, pll b, int len) {
        a.ff = ((a.ff * p[0][len]) + b.ff) % mod[0];
        a.ss = ((a.ss * p[1][len]) + b.ss) % mod[1];
        return (a.ff << 32) | a.ss;
    }

    int cmp(const RabinKarp& other) { // -1 : less, 0 = equal, 1 = greater
        if(n == 0 && other.n == 0) return 0;
        int left = 0, right = min(n, other.n) - 1, res = -1;
        while(left <= right) {
            int middle = midPoint;
            if(get_hash(0, middle + 1) != other.get_hash(0, middle + 1)) res = middle, right = middle - 1;
            else left = middle + 1;
        }
        if(res == -1) {
            return n == other.n ? 0 : (n < other.n ? -1 : 1);
        }
        return s[res] < other.s[res] ? -1 : 1;
    }
};

class MANACHER {    
    public: 
    string s;   
    string ans; 
    string max_prefix, max_suffix;
    ll total_palindrome;
    int n;
    vi man;
    vi prefix; // longest palindrome length starting at index i
    vi suffix; // longest palindrome length ending at index i

    MANACHER(const string s) { 
        total_palindrome = 0;
        this->n = s.size();
        this->s = s;
        build_manacher();
        string odd = get_max_palindrome(s, 1);  
        string even = get_max_palindrome(s, 0);
        ans = odd.size() > even.size() ? odd : even;
        for(int i = 0; i < n; i++) {
            int evenLen = longest_even_palindrome_center_at(i);
            int oddLen = longest_odd_palindrome_center_at(i);
            total_palindrome += (evenLen + 1) / 2 + (oddLen + 1) / 2;
        }
        prefix.assign(n, 1);
        suffix.assign(n, 1);
        int T = man.size();
        for(int i = 0; i < n; ++i) {
            int oddLen = longest_odd_palindrome_center_at(i);  
            if(oddLen > 0) {
                int half = oddLen / 2;
                int L = i - half;
                int R = i + half;
                prefix[L] = max(prefix[L], oddLen);
                suffix[R] = max(suffix[R], oddLen);
            }

            int evenLen = longest_even_palindrome_center_at(i);
            if(evenLen > 0) {
                int half = evenLen / 2;
                int L = i - half + 1;
                int R = i + half;
                if(L >= 0 && R < n) {
                    prefix[L] = max(prefix[L], evenLen);
                    suffix[R] = max(suffix[R], evenLen);
                }
            }
        }
        for(int i = n - 2; i >= 0; --i) {
            suffix[i] = max(suffix[i], suffix[i + 1] - 2);
        }
        for(int i = 1; i < n; ++i) {
            prefix[i] = max(prefix[i], prefix[i - 1] - 2);
        }
        max_prefix = s.substr(0, prefix[0]);
        max_suffix = s.substr(n - suffix.back());
    }

    ll get_total_palindrome() {
        return total_palindrome;
    }
    
    void build_manacher() {
        string t;
        for(char c : s) {
            t.pb('#');
            t.pb(c);
        }
        t.pb('#');
        int T = t.size();
        man.assign(T, 0);
        int L = 0, R = 0;
        for(int i = 0; i < T; i++) {
            if(i < R) {
                man[i] = min(R - i, man[L + R - i]);
            } else {
                man[i] = 0;
            }
            while(i - man[i] >= 0 && i + man[i] < T && t[i - man[i]] == t[i + man[i]]) {
                man[i]++;
            }
            if(i + man[i] > R) {
                L = i - man[i] + 1;
                R = i + man[i] - 1;
            }
        }
    }

    string longest_palindrome() {  
        return ans;
    }

    vi get_manacher(string s, int start) { // odd size palindrome start with 1, even start with 0
        string tmp;
        for(auto& it : s) {
            tmp += "#";
            tmp += it;
        }
        tmp += "#";  
        swap(s, tmp);
        int n = s.size();
        vector<int> p(n); 
        int l = 0, r = 0;  
        for(int i = 0; i < n; i++) {
            if(i < r) {
                p[i] = min(r - i, p[l + r - i]);
            } else {
                p[i] = 0;
            }
            while(i - p[i] >= 0 && i + p[i] < n && s[i - p[i]] == s[i + p[i]]) {
                p[i]++;
            }
            if(i + p[i] > r) {
                l = i - p[i] + 1;
                r = i + p[i] - 1;
            }
        }
        vi result;
        for(int i = start; i < n; i += 2) {
            result.pb(p[i] / 2);
        }
        if(start == 0) { // for even size, shift by one index to the right
            for(int i = 1; i < (int)result.size(); i++) {    
                swap(result[i - 1], result[i]);
            }
            result.pop_back();
        }
        return result;
    }
        
	string get_max_palindrome(const string& s, bool odd) {  
        auto manacher = get_manacher(s, odd);
        int N = manacher.size();
        int start = 0, max_len = 0;
        for(int i = 0; i < N; i++) {    
            int len = manacher[i] * 2 - odd;
            if(len < max_len) continue;
            start = i - manacher[i] + 1;
            max_len = len;
        }
        return s.substr(start, max_len);
    };


    bool is_palindrome(int left, int right) {
        int center = left + right + 1;
        return man[center] >= (right - left + 1);
    }

    int longest_odd_palindrome_center_at(int i) {
        int center = 2 * i + 1;
        if(center >= (int)man.size()) return 0;
        return man[center] - 1;
    }
    
    int longest_even_palindrome_center_at(int i) {
        int center = 2 * i + 2;
        if(center >= (int)man.size()) return 0;
        return man[center] - 1; 
    }
};

template<typename T = string>
struct LCS { // longest common subsequence
    T lcs;
    T shortest_supersequence; // find the shortest string where covers both s and t as subsequence
    LCS(const T& s, const T& t) {
        int n = s.size(), m = t.size();
        vvi dp(n + 1, vi(m + 1));
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= m; j++) {
                if(s[i - 1] == t[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
                else dp[i][j] = max({dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]});
            }
        }
        int curr = dp[n][m];
        for(int i = n; i >= 1; i--) {
            for(int j = m; j >= 1; j--) {
                if(dp[i][j] == curr && s[i - 1] == t[j - 1]) {
                    lcs.pb(s[i - 1]);
                    curr--;
                    break;
                }
            }
        }
        rev(lcs);
        int i = 0, j = 0;
        for(auto& ch : lcs) {
            while(i < n && s[i] != ch) {
                shortest_supersequence.pb(s[i++]);
            }
            while(j < m && t[j] != ch) {
                shortest_supersequence.pb(t[j++]);
            }
            shortest_supersequence.pb(ch);
            i++, j++;
        }
        while(i < n) shortest_supersequence.pb(s[i++]);
        while(j < m) shortest_supersequence.pb(t[j++]);
    }
};

class suffix_array {
    public:
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
    string s;
    int n;
    vi sa, pos, lcp;
    ll distinct_substring;
    linear_rmq<int> rmq;
    suffix_array() {}

    suffix_array(const string& s) {
        this->s = s;
        distinct_substring = 0;
        n = s.size();
        sa.rsz(n), pos.rsz(n), lcp.rsz(n);
        init();
        build_lcp();
        rmq = linear_rmq<int>(lcp, [](const int& a, const int& b) {return a < b;});
    }

    int get_lcp(int i, int j) {
        if(i == j) return s.size() - i;
        i = pos[i], j = pos[j];
        if(i > j) swap(i, j);
        return rmq.query(i, j - 1);
    }

    void sorted_substring(vpii& S) {
        // https://codeforces.com/edu/course/2/lesson/2/5/practice/status
        sort(all(S), [&](const pii &a, const pii& b) {
                    auto& [l1, r1] = a;
                    auto& [l2, r2] = b;
                    int len1 = r1 - l1 + 1;
                    int len2 = r2 - l2 + 1;
                    int common = get_lcp(l1, l2);
                    debug(a, b, common);
                    if(common >= min(len1, len2)) {
                        if(len1 != len2) return len1 < len2;
                        return l1 < l2;
                    }
                    return s[l1 + common] < s[l2 + common];
                });
    }


    void init() {
        vi r(n), tmp(n), sa2(n), cnt(max(256, n) + 1);

        for(int i = 0; i < n; i++) {
            sa[i] = i;
            r[i] = s[i];
        }

        for(int k = 1; k < n; k <<= 1) {
            fill(all(cnt), 0);
            for(int i = 0; i < n; i++) {
                int key = (i + k < n ? r[i + k] + 1 : 0);
                cnt[key]++;
            }
            for(int i = 1; i < (int)cnt.size(); i++) cnt[i] += cnt[i - 1];
            for(int i = n - 1; i >= 0; i--) {
                int j = sa[i];
                int key = (j + k < n ? r[j + k] + 1 : 0);
                sa2[--cnt[key]] = j;
            }

            fill(all(cnt), 0);
            for(int i = 0; i < n; i++) {
                cnt[r[i] + 1]++;
            }
            for(int i = 1; i < (int)cnt.size(); i++) cnt[i] += cnt[i - 1];
            for(int i = n - 1; i >= 0; i--) {
                int j = sa2[i];
                sa[--cnt[r[j] + 1]] = j;
            }

            tmp[sa[0]] = 0;
            for(int i = 1; i < n; i++) {
                auto [a1, b1] = MP(r[sa[i - 1]], sa[i - 1] + k < n ? r[sa[i - 1] + k] : -1);
                auto [a2, b2] = MP(r[sa[i]], sa[i] + k < n ? r[sa[i] + k] : -1);
                tmp[sa[i]] = tmp[sa[i - 1]] + (a1 != a2 || b1 != b2);
            }
            r = tmp;
            if(r[sa[n - 1]] == n - 1) break;  
        }
        for(int i = 0; i < n; i++) {
            pos[sa[i]] = i;
        }
    }

    void build_lcp() {
        for(int i = 0, k = 0; i < n; i++) {
            if(pos[i] == n - 1) continue;
            int j = sa[pos[i] + 1];
            while(s[i + k] == s[j + k]) k++;
            lcp[pos[i]] = k;
            if(k) k--;
        }
        distinct_substring = (ll)n * (n + 1) / 2 - sum(lcp);
    }

    int check(const string& x, int m) {
        int j = sa[m];
        int L = min((int)x.size(), n - j);
        for(int i = 0; i < L; i++) {
            if(s[j + i] < x[i]) return -1;
            if(s[j + i] > x[i]) return  1;
        }
        if((int)x.size() == L) return 0;
        return -1;
    }
     
    pii get_bound(const string& x) {
        int l = 0, r = n - 1, first = -1;
        while(l <= r) {
            int m = (l + r) >> 1;
            int v = check(x, m);
            if(v >= 0) {
                if(v == 0) first = m;
                r = m - 1;
            } else {
                l = m + 1;
            }
        }
        if(first == -1) return {-1, -1};
        l = first; 
        r = n - 1;
        int last = first;
        while(l <= r) {
            int m = (l + r) >> 1;
            int v = check(x, m);
            if(v <= 0) {
                if(v == 0) last = m;
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        return {first, last};
    }

    int count(const string& x) {
        if(x.size() > n) return 0;
        auto [l, r] = get_bound(x);
        return l == -1 ? 0 : r - l + 1;
    }

    string lcs(const string& s, const string& t) {
        string combined = s + '$' + t;
        suffix_array sa_combined(combined);
        int max_lcp = 0, start_pos = 0;
        int split = s.size();
        for(int i = 1; i < sa_combined.n; i++) {
            int suffix1 = sa_combined.sa[i - 1];
            int suffix2 = sa_combined.sa[i];
            bool in_s1 = suffix1 < split;
            bool in_t1 = suffix2 > split;
            bool in_s2 = suffix2 < split;
            bool in_t2 = suffix1 > split;
            if((in_s1 && in_t1) || (in_s2 && in_t2)) {
                int len = sa_combined.lcp[i - 1];
                if(len > max_lcp) {
                    max_lcp = len;
                    start_pos = sa_combined.sa[i];
                }
            }
        }
        return combined.substr(start_pos, max_lcp);
    }

    string kth_distinct(ll k) {
        if(k > (ll)n * (n + 1) / 2) return "";
        ll prev = 0, curr = 0;
        for(int i = 0; i < n; i++) {
            if(curr + (n - sa[i]) - prev >= k) {
                string ans = s.substr(sa[i], prev);
                while(curr < k) {
                    ans += s[sa[i] + prev++];
                    curr++;
                }
                return ans;
            }
            curr += (n - sa[i]) - prev;
            prev = lcp[i];
        }
        return "";
    }

    string lcs(vs& a) {
        int K = a.size();
        if(K == 0) return "";
        if(K == 1) return a[0];

        int total = 0;
        for(auto &s : a) total += s.size() + 1;
        string T; 
        T.reserve(total);
        vi owner;
        owner.reserve(total);
        for(int i = 0; i < K; i++) {
            for(char& c : a[i]) {
                T.pb(c);
                owner.pb(i);
            }
            T.pb(char(1 + i));
            owner.pb(-1);
        }

        suffix_array sa2(T);
        int N2 = sa2.n;

        vi freq(K);
        int have = 0, left = 0;
        int best = 0, bestPos = 0;
        deque<pii> dq;

        for(int right = 0; right < N2; right++) {
            int id = owner[sa2.sa[right]];
            if(id >= 0 && ++freq[id] == 1) have++;

            if(right > 0) {
                int idx = right - 1;
                int v = sa2.lcp[idx];
                while(!dq.empty() && dq.back().ss >= v) dq.pop_back();
                dq.emplace_back(idx, v);
            }

            while(have == K) {
                while(!dq.empty() && dq.front().ff < left) dq.pop_front();
                if(left < right && !dq.empty() && dq.front().ss > best) {
                    best = dq.front().ss;
                    bestPos = sa2.sa[dq.front().ff];
                }
                int idL = owner[sa2.sa[left]];
                if(idL >= 0 && --freq[idL] == 0) have--;
                left++;
            }
        }
        return best > 0 ? T.substr(bestPos, best) : string();
    }

    vi lcp_vector(const string& s, const string& t) { // return a vector for each i in t represents the lcp in s
        int n = s.size(), m = t.size();
        const int N = n + m + 1;
        suffix_array S(s + '#' + t);
        vi prev(N, -1), next(N, -1);
        for(int i = 0; i < N; i++) {
            if(i) prev[i] = prev[i - 1];
            int p = S.sa[i];
            if(p < n) prev[i] = i;
        }
        for(int i = N - 1; i >= 0; i--) {
            if(i < N - 1) next[i] = next[i + 1];
            int p = S.sa[i];
            if(p < n) next[i] = i;
        }
        vi A(m);
        for(int i = n + 1; i < N; i++) {
            int p = S.pos[i];
            int mx = 0;
            if(prev[p] != -1) {
                mx = max(mx, S.get_lcp(i, S.sa[prev[p]]));
            }
            if(next[p] != -1) {
                mx = max(mx, S.get_lcp(i, S.sa[next[p]]));
            }
            A[i - (n + 1)] = mx;
        }
        return A;
    }
};

struct substring_count {
    // https://codeforces.com/contest/914/problem/F
    int n, W;
    string s;
    vt<ull> B[26];

    substring_count(const string& str) : n(str.size()), s(str) {
        W = (n + 63) / 64;
        for(int c = 0; c < 26; c++)
            B[c].assign(W, 0);
        for(int i = 0; i < n; i++) {
            int c = s[i] - 'a';
            int w = i >> 6, b = i & 63;
            B[c][w] |= (1ULL << b);
        }
    }

    void update(int pos, char newc) {
        int oldc = s[pos] - 'a', nc = newc - 'a';
        if(oldc == nc) return;
        int w = pos >> 6, b = pos & 63;
        ull mask = 1ULL << b;
        B[oldc][w] ^= mask;
        B[nc][w] ^= mask;
        s[pos] = newc;
    }

    ll query(int l, int r, const string& y) const {
        int m = y.size();
        if(m > r - l + 1) return 0;
        vt<ull> M = B[y[0] - 'a'];
        for(int j = 1; j < m; j++) {
            int c = y[j] - 'a';
            int sw = j >> 6, sb = j & 63;
            vt<ull> T(W, 0);
            for(int w = 0; w + sw < W; w++) {
                ull low  = B[c][w + sw] >> sb;
                ull high = sb ? B[c][w + sw + 1] << (64 - sb) : 0;
                T[w] = low | high;
            }
            for(int w = 0; w < W; w++)
                M[w] &= T[w];
        }
        int start = l, end = r - m + 1;
        int wL = start >> 6, bL = start & 63;
        int wR = end >> 6, bR = end & 63;
        long long ans = 0;
        if(wL == wR) {
            ull mask = (~0ULL << bL) & (~0ULL >> (63 - bR));
            ans = pct(M[wL] & mask);
        } else {
            ans += pct(M[wL] & (~0ULL << bL));
            for (int w = wL + 1; w < wR; w++)
                ans += pct(M[w]);
            ans += pct(M[wR] & (~0ULL >> (63 - bR)));
        }
        return ans;
    }
};

struct eer_tree {
    struct Node {
        int len, link, next[26];
        ll palindrome;
        Node(int l = 0) : len(l), link(0), palindrome(0) { mset(next, 0); }
    };

    struct SnapInfo {
        int last;
        int max_len;
        int max_end;
        ll total_palindrome;
    };

    vt<Node> F;
    string s;
    int last;
    int max_len, max_end;
    ll total_palindrome;
    vt<SnapInfo> snaps;

    eer_tree() : max_len(0), max_end(0), last(0), total_palindrome(0) {
        init();
    }

    void init(int n = 0) {
        s.clear();
        vt<Node>().swap(F);
        F.reserve(n + 2);
        F.pb(Node(-1));
        F.pb(Node(0));
        total_palindrome = 0;
        last = 1;
        max_len = 0;
        max_end = 1;
        snaps.clear();
    }

    int insert(char ch) {
        s.pb(ch);
        int pos = (int)s.size() - 1;
        int c = ch - 'a';
        int curr = last;
        while(true) {
            int L = F[curr].len;
            if(pos - 1 - L >= 0 && s[pos - 1 - L] == ch) break;
            curr = F[curr].link;
        }
        int created = false;
        if(!F[curr].next[c]) {
            int new_node = (int)F.size();
            F[curr].next[c] = new_node;
            F.pb(Node(F[curr].len + 2));
            if(F.back().len == 1) {
                F.back().link = 1;
                F.back().palindrome = 1;
            } else {
                int link_cand = F[curr].link;
                while(true) {
                    int L = F[link_cand].len;
                    if (pos - 1 - L >= 0 && s[pos - 1 - L] == ch) break;
                    link_cand = F[link_cand].link;
                }
                F.back().link = F[link_cand].next[c];
                F.back().palindrome = F[F.back().link].palindrome + 1;
            }
            created = true;
        }
        last = F[curr].next[c];
        if(F[last].len > max_len) {
            max_len = F[last].len;
            max_end = pos;
        }
        total_palindrome += F[last].palindrome;
        snaps.pb({last, max_len, max_end, total_palindrome});
        return created;
    }

    void pop() {
        if (s.empty()) return;
        s.pop_back();
        snaps.pop_back();
        if(snaps.empty()) {
            last = 1;
            max_len = 0;
            max_end = 1;
            total_palindrome = 0;
        } else {
            auto &st = snaps.back();
            last = st.last;
            max_len = st.max_len;
            max_end = st.max_end;
            total_palindrome = st.total_palindrome;
        }
    }

    string longest_palindrome() {
        int start = max_end - max_len + 1;
        return s.substr(start, max_len);
    }

    int distinct_palindrome() {
        return (int)F.size() - 2;
    }
};

template<int sigma = 26, char mch = 'a'>
struct eertree {
    // https://judge.yosupo.jp/problem/palindromes_in_deque
    eertree(size_t q) {
        q += 2;
        cnt = len = par = link = slink = vector(q, 0);
        to.resize(q);
        link[0] = slink[0] = 1;
        len[1] = -1;
    }

    template<bool back = 1>
    static int get(auto const& d, size_t idx) {
        if(idx >= size(d)) {
            return -1;
        } else if constexpr (back) {
            return prev(end(d))[-idx];
        } else {
            return begin(d)[idx];
        }
    }
    template<bool back = 1>
    static void push(auto &d, auto c) {
        if constexpr (back) {
            d.push_back(c);
        } else {
            d.push_front(c);
        }
    }
    template<bool back = 1>
    static void pop(auto &d) {
        if constexpr (back) {
            d.pop_back();
        } else {
            d.pop_front();
        }
    }

    template<bool back = 1>
    void add_letter(char c) {
        c -= mch;
        push<back>(s, c);
        int pre = get<back>(states, 0);
        int last = make_to<back>(pre, c);
        active += !(cnt[last]++);
        int D = 2 + len[pre] - len[last];
        while(D + len[pre] <= len[last]) {
            pop<back>(states);
            if(!empty(states)) {
                pre = get<back>(states, 0);
                D += get<back>(diffs, 0);
                pop<back>(diffs);
            } else {
                break;
            }
        }
        if(!empty(states)) {
            push<back>(diffs, D);
        }
        push<back>(states, last);
    }
    template<bool back = 1>
    void pop_letter() {
        int last = get<back>(states, 0);
        active -= !(--cnt[last]);
        pop<back>(states);
        pop<back>(s);
        array cands = {pair{link[last], len[last] - len[link[last]]},
                       pair{par[last], 0}};
        for(auto [state, diff]: cands) {
            if(empty(states)) {
                states = {state};
                diffs = {diff};
            } else {
                int D = get<back>(diffs, 0) - diff;
                int pre = get<back>(states, 0);
                if(D + len[state] > len[pre]) {
                    push<back>(states, state);
                    pop<back>(diffs);
                    push<back>(diffs, D);
                    push<back>(diffs, diff);
                }
            }
        }
        pop<back>(diffs);
    }
    void add_letter(char c, bool back = 1) {
        if(back) {
            add_letter<1>(c);
        } else {
            add_letter<0>(c);
        }
    }
    void pop_letter(bool back = 1) {
        if(back) {
            pop_letter<1>();
        } else {
            pop_letter<0>();
        }
    }
    int distinct() { return active; }

    template<bool back = 1>
    int maxlen() { return len[get<back>(states, 0)]; }

    void pop_front() { pop_letter(0); }
    void pop_back() { pop_letter(1); }
    void push_front(char c) { add_letter(c, 0); }
    void push_back(char c) { add_letter(c, 1); }
    int longest_prefix_pal() { return maxlen<0>(); }
    int longest_suffix_pal() { return maxlen<1>(); }

    vector<array<int, sigma>> to;
    vector<int> len, link, slink, par, cnt;

    deque<char> s;
    deque<int> states = {0}, diffs;
    int sz = 2, active = 0;

    template<bool back = 1>
    int get_link(int v, char c) {
        while(c != get<back>(s, len[v] + 1)) {
            if(c == get<back>(s, len[link[v]] + 1)) {
                v = link[v];
            } else {
                v = slink[v];
            }
        }
        return v;
    }

    template<bool back = 1>
    int make_to(int last, char c) {
        last = get_link<back>(last, c);
        if(!to[last][c]) {
            int u = to[get_link<back>(link[last], c)][c];
            link[sz] = u;
            par[sz] = last;
            len[sz] = len[last] + 2;
            if(len[sz] - len[u] == len[u] - len[link[u]]) {
                slink[sz] = slink[u];
            } else {
                slink[sz] = u;
            }
            to[last][c] = sz++;
        }
        return to[last][c];
    }

};



