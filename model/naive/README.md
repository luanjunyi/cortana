##pair analysis of errors

Here is the case by case analysis of wrongly classified domain pairs for the naive bayes model. The notation(d1, d2, c) means there are c cases in the testing data where our model predicted d1 while the actual domain is d2.

### (Web, Note, 73)

    ���� �� ̫ �� �� �� ��ǩ
Word segmentation wrong for '��̫�޶�'. Fixed when seg is right.

    �ҳ� �� �� ���� �� �й� �� ����
    web -58.1265109079 1
    �ҳ�(�ҳ�, 726): -10.2843    ��(��, 245831): -3.9606	��(��, 4604): -7.5232	����(����, 84): -10.0432	��(__LOCATION__, 7924): -2.9422	�й�(�й�, 2004): -9.8490	��(��, 236835): -3.6335	����(����, 6038): -10.8905
    ----------------------------------------------------------------
    note -61.5606162708 1
    �ҳ�(�ҳ�, 726): -8.2810	��(��, 245831): -2.7124	��(��, 4604): -6.9623	����(����, 84): -28.2644	��(__LOCATION__, 7924): -5.1126	�й�(�й�, 2004): -5.3422	��(��, 236835): -2.1340	����(����, 6038): -3.7517
    
�����¡� acutally works pretty well. But the effect got undermined by other words like '����', which got much lower probability yet larger difference among them. 

I tried to use sqrt to smooth the probabilities. CV accuracy is unaffected but testing accuracy got went up to 87.2%. However there are other cases like 

    ���� �� �� �� �� ��

Where �����졯 and ����' favors weather domain, yet ���֡� and other words flooded too much noise.
    
    