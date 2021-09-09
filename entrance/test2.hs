brackets :: [Char] -> Bool
brackets list = brackets' list "" where
	brackets' "" ""		= True
	brackets' "" (y:ys)
		| elem y "([{"	= False
		| otherwise		= brackets' "" ys
	brackets' (x:xs) ""
		| elem x ")]}"	= False
		| elem x "([{"	= brackets' xs [x]
		| otherwise		= brackets' xs ""
	brackets' (x:xs) (y:ys)
		| elem x ")]}"	= if (elem (y, x) [('(', ')'),('[', ']'),('{', '}')]) then (brackets' xs ys) else False
		| elem x "([{"	= brackets' xs (x:y:ys)
		| otherwise		= brackets' xs (y:ys)
