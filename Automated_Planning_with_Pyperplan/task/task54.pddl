(define (problem BLOCKS-7-0)
(:domain BLOCKS)
(:objects A B C D E F G - block)
(:INIT (CLEAR A) (CLEAR G) (ONTABLE B) (ONTABLE C) (ON A B) (ON G F) (ON F E) (ON E D) (ON D C) (HANDEMPTY))
(:goal (AND (ON G F) (ON F E) (ON E D) (ON D C) (ON C B) (ON B A)))
)