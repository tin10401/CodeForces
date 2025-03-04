int T[MX * MK][2], cnt[MX * MK], ptr;
class Binary_Trie { 
    public:
    int m = 20;
    void insert(ll num, int v = 1) {  
        int curr = 0;   
        for(int i = m - 1; i >= 0; i--) {  
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
        for(int i = m - 1; i >= 0; i--) {  
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
        for(int i = m - 1; i >= 0; i--) {  
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
        for(int i = m - 1; i >= 0; i--) {
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
		// res += cnt[curr] -> count equal to, remove if needed
        return res;
    }
	
	ll count_greater_than(ll a, ll b) {
        int curr = 0;
        ll res = 0;
        for(int i = m - 1; i >= 0; i--) {
            int bits = (a >> i) & 1;
            int b_bits = (b >> i) & 1;
            if(b_bits == 0 && T[curr][!bits]) {
                res += cnt[T[curr][!bits]];
            }
            curr = T[curr][b_bits ^ bits];
            if(!curr) break;
        }
        // res += cnt[curr]; -> counter equal to, remove if needed
        return res;
    }

	
	ll find_mex(ll x) { // find a first missing number
        ll mex = 0, curr = 0;
        for(int i = m - 1; i >= 0; i--) {
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
    }
    ptr = 0;
}

struct TrieNode
{
    TrieNode* sfx, *dict, *children[26];
    int id = -1;
};

static TrieNode nodes[(int)5e4 + 1];

class Trie
{
    public:
    TrieNode* root;
    int count = 0;
	
	TrieNode* newTrieNode() {
        nodes[count] = TrieNode();
        return &nodes[count++];
    }
	
	Trie() {    
        count = 0;
        root = newTrieNode();   
    }

//    Trie(vector<string>& words, vector<int>& costs, string target) {
//        root = newTrieNode();
//        root->sfx = root->dict = root;
//
//    }

    
    void insert(const string& s) {  
        TrieNode* curr = root;
        for(auto& ch : s) {
            if(!curr->children[ch - 'a']) curr->children[ch - 'a'] = newTrieNode();
            curr = curr->children[ch - 'a'];
        }

    }
    
    void aho_corasick() {   
        queue<TrieNode*> q;
        q.push(root);
        while(!q.empty()) {
            TrieNode* par = q.front();
            q.pop();
            for(int i = 0; i < 26; i++) {
                TrieNode* child = par->children[i];
                if(!child) continue;
                TrieNode* suff = par->sfx;
                while(suff != root && !suff->children[i]) suff = suff->sfx;
                if(par != root && suff->children[i]) child->sfx = suff->children[i];
                else child->sfx = root;

                child->dict = child->sfx->id == -1 ? child->sfx->dict : child->sfx;
                q.push(child);
            }
        }
    }

    void queries(TrieNode*& prev, int i, char ch)
    {
        while(prev != root && !prev->children[ch - 'a']) prev = prev->sfx;
        if(prev->children[ch - 'a']) {
            prev = prev->children[ch - 'a'];
            TrieNode* curr = prev->id == -1 ? prev->dict : prev;
            while(curr->id != -1){
                int j = curr->id;
                curr = curr->dict;
            }
        }
    }
};

vi KMP(const string& s) {   
    int n = s.size();
    vi prefix(n);
    for(int i = 1, j = 0; i < n; i++) { 
        while(j && s[i] != s[j]) j = prefix[j - 1]; 
        if(s[i] == s[j]) prefix[i] = ++j;
    }
    return prefix;
	// property of finding period by kmp : if(len % (len - kmp[i]) == 0) period = len - kmp[i], otherwise period = len
	
//    int n = s.size(); // for jumping with large size of s
//    s = ' ' + s;
//    vi prefix(n + 1);   
//    vvi dp(n + 1, vi(26));  
//    if(n >= 2) dp[1][s[2] - 'a'] = 1;
//    for(int i = 2, j = 0; i <= n; i++) {   
//        while(j && s[i] != s[j + 1]) j = prefix[j]; 
//        if(s[i] == s[j + 1]) j++;   
//        prefix[i] = j;  
//        dp[i] = dp[j];
//        if(i < n) dp[i][s[i + 1] - 'a'] = i;
//    }
//    int q; cin >> q;
//    while(q--) {    
//        string t; cin >> t; 
//        int m = t.size();   
//        s += t; 
//        for(int i = n + 1, j = prefix[n]; i <= n + m; i++) {   
//            while(j > n && s[i] != s[j + 1]) j = prefix[j];
//            if(j && s[j + 1] != s[i]) j = dp[j][s[i] - 'a'];
//            if(s[j + 1] == s[i]) j++;   
//            prefix.pb(j);   
//            cout << j << ' '; 
//        }
//        cout << endl;   
//        for(int i = 0; i < m; i++) {    
//            prefix.pop_back();  
//            s.pop_back();
//        }
//    }


}

int count_substring(const string& s, const string& t) { // s is main string, t is pattern
    auto kmp = KMP(t);
    int N = s.size(), M = t.size();
    int cnt = 0;
//    vi occur;
    for(int i = 0, j = 0; i < N;) {
        if(s[i] == t[j]) i++, j++;
        else if(j) j = kmp[j - 1];
        else i++;
        if(j == M) {
//            occur.pb(i - M);
//            j = kmp[j - 1];
            cnt++;
            j = 0;
        }
    }
    return cnt;
}

vi Z_Function(const string& s) {    
    int n = s.size();   
    vi prefix(n);   
    for(int i = 1, left = 0, right = 0; i < n; i++) {   
        if(i > right) { 
            left = right = i;   
            while(right < n && s[right] == s[right - left]) right++;    
            prefix[i] = right-- - left;
        }
        else {  
            if(prefix[i - left] + i < right + 1) {  
                prefix[i] = prefix[i - left];
            }
            else {  
                left = i;   
                while(right < n && s[right] == s[right - left]) right++;    
                prefix[i] = right-- - left;
            }
        }
    }
    return prefix;
}

const int HASH_COUNT = 2;
vll globalBase;
vll globalMod;
void initGlobalHashParams() {
    if (!globalBase.empty() && !globalMod.empty()) return;
    vll candidateBases = {29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    vll candidateMods  = {1000000007LL, 1000000009LL, 1000000021LL, 1000000033LL,
                                 1000000087LL, 1000000093LL, 1000000097LL, 1000000103LL};
								 
	unsigned seed = chrono::steady_clock::now().time_since_epoch().count();
    shuffle(candidateBases.begin(), candidateBases.end(), default_random_engine(seed));
    shuffle(candidateMods.begin(), candidateMods.end(), default_random_engine(seed + 1));

    globalBase.rsz(HASH_COUNT);
    globalMod.rsz(HASH_COUNT);
    for (int i = 0; i < HASH_COUNT; i++) {
        globalBase[i] = candidateBases[i];
        globalMod[i]  = candidateMods[i];
    }
}
template<class T = string>
struct RabinKarp {
    vvll prefix, pow;
    int n;
    
    RabinKarp(const T &s) {
        initGlobalHashParams();
        n = s.size();
        prefix.rsz(HASH_COUNT);
        pow.rsz(HASH_COUNT);
        for (int i = 0; i < HASH_COUNT; i++) {
            prefix[i].rsz(n + 1, 0);
            pow[i].rsz(n + 1, 1);
        }
        buildHash(s);
    }
    
    void buildHash(const T &s) {
        for (int j = 1; j <= n; j++) {
            int x = s[j - 1] - 'a' + 1;
            for (int i = 0; i < HASH_COUNT; i++) {
                prefix[i][j] = (prefix[i][j - 1] * globalBase[i] + x) % globalMod[i];
                pow[i][j] = (pow[i][j - 1] * globalBase[i]) % globalMod[i];
            }
        }
    }
    
    ll get_hash(int l, int r) {
        if (l < 0 || r > n || l > r) return 0;
        ll hash0 = prefix[0][r] - (prefix[0][l] * pow[0][r - l] % globalMod[0]);
        hash0 = (hash0 % globalMod[0] + globalMod[0]) % globalMod[0];
        ll hash1 = prefix[1][r] - (prefix[1][l] * pow[1][r - l] % globalMod[1]);
        hash1 = (hash1 % globalMod[1] + globalMod[1]) % globalMod[1];
        return (hash0 << 32) | hash1;
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
        a.ff = ((a.ff * pow[0][len]) + b.ff) % globalMod[0];
        a.ss = ((a.ss * pow[1][len]) + b.ss) % globalMod[1];
        return (a.ff << 32) | a.ss;
    }
};

class MANACHER {    
    public: 
    string s;   
    string ans; 
    ll total_palindrome;
    int n;
    vi man;
    MANACHER(const string& s) { 
        total_palindrome = 0;
        this->n = s.size();
        this->s = s;
        build_manacher();
        string odd = get_max_palindrome(s, 1);  
        string even = get_max_palindrome(s, 0);
        ans = odd.size() > even.size() ? odd : even;
        for(int i = 0; i < n; i++) {
            int even = longest_even_palindrome_at(i);
            int odd = longest_odd_palindrome_at(i);
            total_palindrome += (even + 1) / 2 + (odd + 1) / 2;
        }
    }

    ll get_total_palindrome() {
        return total_palindrome;
    }
    
    void build_manacher() {
        string t;
        for (char c : s) {
            t.push_back('#');
            t.push_back(c);
        }
        t.push_back('#');
        int T = t.size();
        man.assign(T, 0);
        int L = 0, R = 0;
        for (int i = 0; i < T; i++) {
            if (i < R) {
                man[i] = min(R - i, man[L + R - i]);
            } else {
                man[i] = 0;
            }
            while (i - man[i] >= 0 && i + man[i] < T && t[i - man[i]] == t[i + man[i]]) {
                man[i]++;
            }
            if (i + man[i] > R) {
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
        for (auto& it : s) {
            tmp += "#";
            tmp += it;
        }
        tmp += "#";  
        swap(s, tmp);
        int n = s.size();
        vector<int> p(n); 
        int l = 0, r = 0;  
        for (int i = 0; i < n; i++) {
            if (i < r) {
                p[i] = min(r - i, p[l + r - i]);
            } else {
                p[i] = 0;
            }
            while (i - p[i] >= 0 && i + p[i] < n && s[i - p[i]] == s[i + p[i]]) {
                p[i]++;
            }
            if (i + p[i] > r) {
                l = i - p[i] + 1;
                r = i + p[i] - 1;
            }
        }
        vi result;
        for (int i = start; i < n; i += 2) {
            result.push_back(p[i] / 2);
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
//            if(i - manacher[i] + 1 == 0) {  // max prefix_palindrome
//                max_len = len;  
//                start = 0;
//            }
//            else if(i + manacher[i] + !odd == N) { // max_suffix_palindrome
//                max_len = len;  
//                start = i - manacher[i] + 1;
//            }
            start = i - manacher[i] + 1; // max_palindrome overall
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

struct LCS { // longest common subsequence
    string lcs;
    string shortest_supersequence; // find the shortest string where covers both s and t as subsequence
    LCS(const string& s, const string& t) {
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
                    lcs += s[i - 1];
                    curr--;
                    break;
                }
            }
        }
        rev(lcs);
        int i = 0, j = 0;
        for(auto& ch : lcs) {
            while(i < n && s[i] != ch) {
                shortest_supersequence += s[i++];
            }
            while(j < m && t[j] != ch) {
                shortest_supersequence += t[j++];
            }
            shortest_supersequence += ch;
            i++, j++;
        }
        while(i < n) shortest_supersequence += s[i++];
        while(j < m) shortest_supersequence += t[j++];
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


