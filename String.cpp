int T[MX * MK][2], cnt[MX * MK], ptr;
class Binary_Trie { 
    public:
    int m = 30;
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
        int b = (num >> bit) & 1;
        if(!T[curr][b]) T[curr][b] = ++ptr;
        int nxt = T[curr][b];
        if(bit == 0) cnt[nxt] = 1;
        else dfs_insert(nxt, num, bit - 1);
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
                res += cnt[T[curr][bits]];
                curr = T[curr][!bits];
            }
            else {
                curr = T[curr][bits];
            }
            if(!curr) break;
        }
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

class RabinKarp {   
    public: 
    vvll prefix, pow;
    vll base, mod;
    int n, m;
    RabinKarp(const string& s) {  
        m = 3;
        base = {26, 28, 30};    
        mod = {(int)1e9 + 7, (int)1e9 + 33, (int)1e9 + 73};
        n = s.size(); 
        pow.rsz(m), prefix.rsz(m); 
        for(int i = 0; i < m; i++) {    
            pow[i].rsz(n + 1, 1);  
            prefix[i].rsz(n + 1);
        }
        buildHash(s);
    }
    
    void buildHash(const string& s) {   
        for(int j = 1; j <= n; j++) {   
            int x = s[j - 1] - 'a' + 1;
            for(int i = 0; i < m; i++) {    
                prefix[i][j] = (prefix[i][j - 1] * base[i] + x) % mod[i];   
                pow[i][j] = (pow[i][j - 1] * base[i]) % mod[i];
            }
        }
    }
    
    int getHash(int l, int r) { 
		if(l < 0 || r > n || l > r) return -1;
        int hash = prefix[0][r] - (prefix[0][l] * pow[0][r - l] % mod[0]) % mod[0];
        hash = (hash + mod[0]) % mod[0];
        return hash;
		
//        if(l < 0 || r > n || l > r) return {-1, -1};
//        vll ans;    
//        for(int i = 0; i < m; i++) {    
//            ll hash = prefix[i][r] - (prefix[i][l] * pow[i][r - l] % mod[i]) % mod[i]; 
//            hash = (hash + mod[i]) % mod[i];
//            ans.pb(hash);
//        }
//        return MP(ans[0], ans[1]);
    };

	bool diff_by_one_char(RabinKarp& a, int offSet = 0) { // a.size() > n
        int left = 0, right = n, rightMost = -1;    
        while(left <= right) {  
            int middle = midPoint;  
            if(a.getHash(offSet, middle + offSet) == getHash(0, middle)) rightMost = middle, left = middle + 1; 
            else right = middle - 1;
        }
        return a.getHash(rightMost + 1 + offSet, offSet + n) == getHash(rightMost + 1, n);
    }

};

class MANACHER {    
    public: 
    string s;   
    string ans; 
    RabinKarp a, b;
    int n;
    MANACHER(const string& s) : a(s), b(string(s.rbegin(), s.rend())) { 
        this->n = s.size();
        string odd = get_max_palindrome(s, 1);  
        string even = get_max_palindrome(s, 0);
        ans = odd.size() > even.size() ? odd : even;
    }
    
    string get() {  
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
            if(i - manacher[i] + 1 == 0) {  // max prefix_palindrome
                max_len = len;  
                start = 0;
            }
            else if(i + manacher[i] + !odd == N) { // max_suffix_palindrome
                max_len = len;  
                start = i - manacher[i] + 1;
            }
            //start = i - manacher[i] + 1; // max_palindrome overall
            //max_len = len;
        }
        return s.substr(start, max_len);
    };

    bool is_palindrome(int left, int right) {
        int rev_left = n - right - 1, rev_right = n - left - 1;
        return a.getHash(left, right) == b.getHash(rev_left, rev_right);
    }
};

int lcs(const string& s, const string& t) { // longest common subsequences
    int n = s.size(), m = t.size(); 
    vvi dp(n + 1, vi(m + 1));
    for(int i = 1; i <= n; i++) {   
        for(int j = 1; j <= m; j++) {   
            if(s[i - 1] == t[j - 1]) {  
                dp[i][j] = dp[i - 1][j - 1] + 1;
            }
            else {  
                dp[i][j] = max({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
            }
        }
    }
    //return n + m - 2 * dp[n][m];
    return dp[n][m];
}


