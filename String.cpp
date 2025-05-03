int T[MX * MK][2], cnt[MX * MK], ptr;
class Binary_Trie { 
    public:
    void insert(ll num, int v = 1) {  
        int curr = 0;   
        for(int i = MK - 1; i >= 0; i--) {  
            int bits = (num >> i) & 1;  
            if(!T[curr][bits]) T[curr][bits] = ++ptr, cnt[ptr] = 0;   
            curr = T[curr][bits];
			cnt[curr] += v;
        }
		// dfs_insert(0, num, m - 1);
    }
	
	void dfs_insert(int curr, ll num, int bit) {
		if(bit == -1) {
			cnt[curr ] = 1;
			return;
		}
        int b = (num >> bit) & 1;
        if(!T[curr][b]) T[curr][b] = ++ptr;
        int nxt = T[curr][b];
        dfs_insert(nxt, num, bit - 1);
        cnt[curr] = cnt[nxt] + (T[curr][!b] ? cnt[T[curr][!b]] : 0);
    }

        
    ll max_xor(ll num) {  
        ll res = 0, curr = 0;
        for(int i = MK - 1; i >= 0; i--) {  
            int bits = (num >> i) & 1;  
            if(T[curr][!bits] && cnt[T[curr][!bits]]) {    
                curr = T[curr][!bits];
                res |= (1LL << i);
            }
            else {  
                curr = T[curr][bits];
            }
            if(!curr) break;
        }
        return res;
    }
        
    ll min_xor(ll num) {  
        ll res = num, curr = 0;
        for(int i = MK - 1; i >= 0; i--) {  
            int bits = (num >> i) & 1;  
            if(T[curr][bits] && cnt[T[curr][bits]]) {    
                curr = T[curr][bits];
                if(bits) res ^= (1LL << i);
            }
            else {  
                curr = T[curr][!bits];
                if(!bits) res ^= (1LL << i);
            }
            if(!curr) break;
        }
        return res;
    }
	
	ll count_less_than(ll a, ll b) {
        int curr = 0;
        ll res = 0;
        for(int i = MK - 1; i >= 0; i--) {
            int bits = (a >> i) & 1;
            int b_bits = (b >> i) & 1;
            if(b_bits) {
				if(T[curr][bits]) {
					res += cnt[T[curr][bits]];
				}
                curr = T[curr][!bits];
            }
            else {
                curr = T[curr][bits];
            }
            if(!curr) break;
        }
		// res += cnt[curr]; // remove comments if count equal to as well
        return res;
    }
	
	ll count_greater_than(ll a, ll b) {
        int curr = 0;
        ll res = 0;
        for(int i = MK - 1; i >= 0; i--) {
            int bits = (a >> i) & 1;
            int b_bits = (b >> i) & 1;
            if(b_bits == 0 && T[curr][!bits]) {
                res += cnt[T[curr][!bits]];
            }
            curr = T[curr][b_bits ^ bits];
            if(!curr) break;
        }
		// res += cnt[curr]; // remove comments if count equal to as well
        return res;
    }

	
	ll find_mex(ll x) { // find a first missing number
        ll mex = 0, curr = 0;
        for(int i = MK - 1; i >= 0; i--) {
            int bit = (x >> i) & 1;
            int c = T[curr][bit] ? cnt[T[curr][bit]] : 0;
            if(c < (1LL << i)) {
                curr = T[curr][bit];
            }
            else {
                mex |= (1LL << i);
                curr = T[curr][!bit];
            }
            if(!curr) break;
        }
        return mex;
    }
};

void reset() {  
    for(int i = 0; i <= ptr; i++) { 
        T[i][0] = T[i][1] = 0;
        cnt[i] = 0;
    }
    ptr = 0;
}

const int MM = MX * MK * 3;
int root[MX], T[MM][2], ptr, cnt[MM], level[MM];
struct PERSISTENT_TRIE {
    int insert(int prev, int num, int v = 1, int lev = 0) {   
        int newRoot = ++ptr;    
        int curr = newRoot;
        for(int i = MK - 1; i >= 0; i--)    {   
            int bits = (num >> i) & 1;  
            T[curr][!bits] = T[prev][!bits];
            T[curr][bits] = ++ptr;  
            prev = T[prev][bits];   
            curr = T[curr][bits];   
            level[curr] = lev;
            cnt[curr] = cnt[prev];
            cnt[curr] += v;
        }
        return newRoot;
    }

    int max_xor(int curr, int num, int lev = 0) {
        int res = 0;
        for(int i = MK - 1; i >= 0; i--) {
            int bits = (num >> i) & 1;
            int nxt = T[curr][!bits];
            if(nxt && cnt[nxt] && level[nxt] >= lev) {
                res |= 1LL << i;
                curr = nxt;
            } else {
                curr = T[curr][bits];
            }
            if(!curr) break;
        }
        return res;
    }

    int min_xor(int curr, int num, int lev = 0) {
        int res = num;
        for(int i = MK - 1; i >= 0; i--) {
            int bits = (num >> i) & 1;
            int nxt = T[curr][bits];
            if(nxt && cnt[nxt] && level[nxt] >= lev) {
                curr = nxt;
                if(bits) res ^= 1LL << i;
            }
            else {
                curr = T[curr][!bits];
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
        for(int i = MK - 1; i >= 0; i--) {
            int bits = (x >> i) & 1;
            int same_count = 0;
            for(auto& [l, r] : curr) {
                same_count += (cnt[T[r][bits]] - cnt[T[l][bits]]);
            }
            if(same_count >= k) {
                for(auto& [l, r] : curr) {
                    l = T[l][bits];
                    r = T[r][bits];
                }
                continue;
            }
            k -= same_count;
            for(auto& [l, r] : curr) {
                l = T[l][!bits];
                r = T[r][!bits];
            }
            res |= 1LL << i;
        } 
        return res;
    }
};

void reset() {  
    for(int i = 0; i <= ptr; i++) { 
        T[i][0] = T[i][1] = 0;
        cnt[i] = 0;
        level[i] = 0;
        if(i < MX) root[i] = 0;
    }
    ptr = 0;
}

const int MM = MX * 26;
int T[MM][26], ptr, cnt[MM], ending[MM];
struct Trie {
    bool is_character;
    Trie(bool is_character = true) : is_character(is_character) {}

    int get(char c) {
        return c - (is_character ? 'a' : '0');
    }

    void insert(const string& s, int v = 1) {
        int curr = 0;
        for(auto& ch : s) {
            int j = get(ch);
            if(!T[curr][j]) T[curr][j] = ++ptr;
            curr = T[curr][j];
            cnt[curr] += v;
        }
        ending[curr] = true;
    }

    int count_word_prefix(const string& s) { // how many word is s a prefix of
        int curr = 0;
        for(auto& ch : s) {
            curr = T[curr][get(ch)];
            if(!curr || !cnt[curr]) break;
        }
        return curr ? cnt[curr] : 0;
    }

    int get_max_length_prefix(const string& s) { // get max lcp s[i], s[j] where i != j
        int curr = 0, res = 0;
        for(auto& ch : s) {
            curr = T[curr][get(ch)];
            if(!curr || !cnt[curr]) break;
            res++;
        }
        return res;
    }
};

void reset() {
    for(int i = 0; i <= ptr; i++) {
        mset(T[i], 0);
        cnt[i] = 0;
        ending[i] = 0;
    }
    ptr = 0;
}

class AHO {
public:
    Trie* trie;
    AHO(Trie* t) { trie = t; }
    void build() {
        queue<TrieNode*> q;
        trie->root->sfx = trie->root;
        trie->root->dict = trie->root;
        q.push(trie->root);
        while(!q.empty()){
            TrieNode* par = q.front();
            q.pop();
            for(int i = 0; i < 26; i++){
                TrieNode* child = par->children[i];
                if(!child) continue;
                TrieNode* suff = par->sfx;
                while(suff != trie->root && !suff->children[i])
                    suff = suff->sfx;
                if(par != trie->root && suff->children[i])
                    child->sfx = suff->children[i];
                else
                    child->sfx = trie->root;
                child->dict = (child->sfx->id == -1 ? child->sfx->dict : child->sfx);
                q.push(child);
            }
        }
    }
    void query(TrieNode*& prev, int i, char ch) {
        while(prev != trie->root && !prev->children[ch - 'a'])
            prev = prev->sfx;
        if(prev->children[ch - 'a']){
            prev = prev->children[ch - 'a'];
            TrieNode* curr = (prev->id == -1 ? prev->dict : prev);
            while(curr->id != -1){
                int j = curr->id;
                curr = curr->dict;
            }
        }
    }
};

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
        10007ULL,10009ULL,10037ULL,10039ULL,10061ULL,10067ULL,10069ULL,10079ULL,10091ULL,10093ULL,
        10099ULL,10103ULL,10111ULL,10133ULL,10139ULL,10141ULL,10151ULL,10159ULL,10163ULL,10169ULL,
        10177ULL,10181ULL,10193ULL,10211ULL,10223ULL,10243ULL,10247ULL,10253ULL,10259ULL,10267ULL,
        10271ULL,10273ULL,10289ULL,10301ULL,10303ULL,10313ULL,10321ULL,10331ULL,10333ULL,10337ULL,
        10343ULL,10357ULL,10369ULL,10391ULL,10399ULL,10427ULL,10429ULL,10433ULL,10453ULL,10457ULL,
        10459ULL,10463ULL,10477ULL,10487ULL,10499ULL,10501ULL,10513ULL,10529ULL,10531ULL,10559ULL,
        10567ULL,10589ULL,10597ULL,10601ULL,10607ULL,10613ULL,10627ULL,10631ULL,10639ULL,10651ULL,
        10657ULL,10663ULL,10667ULL,10687ULL,10691ULL,10709ULL,10711ULL,10723ULL,10729ULL,10733ULL,
        10739ULL,10753ULL,10771ULL,10781ULL,10789ULL,10799ULL,10831ULL,10837ULL,10847ULL,10853ULL,
        10859ULL,10861ULL,10867ULL,10883ULL,10889ULL,10891ULL,10903ULL,10909ULL
    };
    vll candidateMods = {
        1000000007ULL,1000000009ULL,1000000033ULL,1000000087ULL,1000000093ULL,
        1000000097ULL,1000000103ULL,1000000123ULL,1000000181ULL,1000000207ULL,
        1000000223ULL,1000000241ULL,1000000271ULL,1000000289ULL,1000000297ULL,
        1000000321ULL,1000000349ULL,1000000363ULL,1000000403ULL,1000000409ULL,
        1000000411ULL,1000000427ULL,1000000433ULL,1000000439ULL,1000000447ULL,
        1000000453ULL,1000000459ULL,1000000483ULL,1000000513ULL,1000000531ULL,
        1000000579ULL,1000000607ULL,1000000613ULL,1000000637ULL,1000000663ULL,
        1000000711ULL,1000000753ULL,1000000787ULL,1000000801ULL,1000000829ULL,
        1000000861ULL,1000000871ULL,1000000891ULL,1000000901ULL,1000000919ULL,
        1000000931ULL,1000000933ULL,1000000993ULL,1000001011ULL,1000001021ULL,
        1000001053ULL,1000001087ULL,1000001089ULL,1000001107ULL,1000001163ULL,
        1000001171ULL,1000001193ULL,1000001201ULL,1000001231ULL,1000001269ULL,
        1000001283ULL,1000001311ULL,1000001327ULL,1000001363ULL,1000001371ULL,
        1000001381ULL,1000001413ULL,1000001431ULL,1000001471ULL,1000001501ULL,
        1000001531ULL,1000001581ULL,1000001613ULL,1000001637ULL,1000001663ULL,
        1000001671ULL,1000001693ULL,1000001703ULL,1000001733ULL,1000001741ULL,
        1000001781ULL,1000001801ULL,1000001863ULL,1000001891ULL,1000001903ULL,
        1000001911ULL,1000001931ULL,1000001933ULL,1000001971ULL,1000001981ULL,
        1000002021ULL,1000002071ULL,1000002083ULL,1000002101ULL,1000002133ULL
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
    vll prefix[HASH_COUNT], suffix[HASH_COUNT];
    int n;
    
    RabinKarp() : n(0) {
        for(int i = 0; i < HASH_COUNT; i++) {
            prefix[i].pb(0);
            suffix[i].pb(0);
        }
    }
    RabinKarp(const T &s) {
        n = s.size();
        for (int i = 0; i < HASH_COUNT; i++) {
            prefix[i].rsz(n + 1, 0);
            suffix[i].rsz(n + 1, 0);
        }
        for (int j = 1; j <= n; j++) {
            int x = s[j - 1] - 'a';
            int y = s[n - j] - 'a';
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
        for (int i = 0; i < n; i++) {
            int evenLen = longest_even_palindrome_at(i);
            int oddLen = longest_odd_palindrome_at(i);
            total_palindrome += (evenLen + 1) / 2 + (oddLen + 1) / 2;
        }
        prefix.assign(n, 1);
        suffix.assign(n, 1);
        int T = man.size(); 
        for(int c = 0; c < T; c++) {
            if(man[c] <= 1) continue;
            if(c % 2 == 1) { 
                int i = (c - 1) / 2;
                int len = man[c] - 1;
                int half = len / 2;
                int L = i - half;
                int R = i + half;
                if (L >= 0 && R < n) {
                    prefix[L] = max(prefix[L], len);
                    suffix[R] = max(suffix[R], len);
                }
            } else { 
                if (c == 0 || c == T - 1) continue;
                int i = (c - 2) / 2;
                int len = man[c] - 1; 
                int half = (len - 1) / 2;
                int L = i - half;
                int R = i + 1 + half;
                if (L >= 0 && R < n) {
                    prefix[L] = max(prefix[L], len);
                    suffix[R] = max(suffix[R], len);
                }
            }
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

    int longest_odd_palindrome_at(int i) {
        int center = 2 * i + 1;
        if (center >= (int)man.size()) return 0;
        return man[center] - 1;
    }
    
    int longest_even_palindrome_at(int i) {
        int center = 2 * i + 2;
        if (center >= (int)man.size()) return 0;
        return man[center] - 1; 
    }
};

class suffix_array {
    public:
    string s;
    int n;
    vi sa, pos, lcp;
    ll distinct_substring;
    suffix_array(const string& s) {
        this->s = s;
        distinct_substring = 0;
        n = s.size();
        sa.rsz(n), pos.rsz(n), lcp.rsz(n);
        init();
        build_lcp();
    }

    void init() {
        vi tmp(n);
        for(int i = 0; i < n; i++) {
            sa[i] = i;
            pos[i] = s[i];
        }
        for(int gap = 1; ; gap <<= 1) {
            auto cmp = [&](int x, int y) -> bool {
                if(pos[x] != pos[y]) return pos[x] < pos[y];
                x += gap, y += gap;
                return x < n && y < n ? pos[x] < pos[y] : x > y;
            };
            sort(all(sa), cmp);
            for(int i = 0; i < n - 1; i++) {
                tmp[i + 1] = tmp[i] + cmp(sa[i], sa[i + 1]);
            }
            for(int i = 0; i < n; i++) pos[sa[i]] = tmp[i];
            if(tmp[n - 1] == n - 1) break;
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
        int found = -1, j = sa[m];
        if(n - j >= (int)x.size()) found = 0;
        for(int i = 0; i < min(n, (int)x.size()); i++)
        {
            if(s[i + j] < x[i]) return -1;
            if(s[i + j] > x[i]) return 1;
        }
        return found;
    }

    int count(const string& x) {
        int left = 0, right = n - 1, left_most = -1, right_most = -1;
        while(left <= right) {
            int middle = midPoint;
            int val = check(x, middle);
            if(val == 0) left_most = middle, right = middle - 1;
            else if(val == -1) left = middle + 1;
            else right = middle - 1;
        }
        if(left_most == -1) return 0;
        left = left_most, right = n - 1;
        while(left <= right) {
            int middle = midPoint;
            int val = check(x, middle);
            if(val == 0) right_most = middle, left = middle + 1;
            else if(val == -1) left = middle + 1;
            else right = middle - 1;
        }
        return right_most - left_most + 1;
    }

    string lcs(const string& t) {
        string combined = s + '$' + t;
        suffix_array sa_combined(combined);
        int max_lcp = 0, start_pos = 0;
        int split = s.size();
        for (int i = 1; i < sa_combined.n; i++) {
            int suffix1 = sa_combined.sa[i - 1];
            int suffix2 = sa_combined.sa[i];
            bool in_s1 = suffix1 < split;
            bool in_t1 = suffix2 > split;
            bool in_s2 = suffix2 < split;
            bool in_t2 = suffix1 > split;
            if ((in_s1 && in_t1) || (in_s2 && in_t2)) {
                int len = sa_combined.lcp[i - 1];
                if (len > max_lcp) {
                    max_lcp = len;
                    start_pos = sa_combined.sa[i];
                }
            }
        }
        return combined.substr(start_pos, max_lcp);
    }
};


