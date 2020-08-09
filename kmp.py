class Solution(object):
    def shortestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        n = len(s)
        if n <= 1:
            return s
        tmp = s + '#' + s[::-1]
        k = 0
        next = [0 for i in range(len(tmp))]
        for i in range(1, len(tmp)):
            while k > 0 and tmp[i] != tmp[k]:
                k = next[k-1]
            if tmp[i] == tmp[k]:
                k += 1
            next[i] = k
        return s[:next[-1]-1:-1] + s


if __name__ == "__main__":
    a = Solution()
    s = "helloword"
    b = a.shortestPalindrome(s)
    print(b)
