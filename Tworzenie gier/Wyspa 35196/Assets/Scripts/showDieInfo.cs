using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class showDieInfo : MonoBehaviour
{
    bool showInfo = false;

    // Start is called before the first frame update
    void Start()
    {
        // Pocz�tkowo ukryj obiekt na podstawie warto�ci showInfo
        SetObjectVisibility();
    }

    // Update is called once per frame
    void Update()
    {
        // Przyk�adowe warunki zmiany warto�ci showInfo (mo�esz dostosowa� je wed�ug potrzeb)
        if (Input.GetKeyDown(KeyCode.Space))
        {
            showInfo = !showInfo;
            SetObjectVisibility();
        }
    }

    void SetObjectVisibility()
    {
        // Sprawd�, czy obiekt ma komponent Collider (lub inny komponent, kt�ry m�g�by go renderowa�)
        Collider collider = GetComponent<Collider>();

        // Je�li collider istnieje, mo�esz ukry� obiekt przez ustawienie aktywno�ci komponentu
        if (collider != null)
        {
            collider.enabled = showInfo;
        }
        else
        {
            // Je�li collider nie istnieje, u�yj aktywno�ci samego obiektu
            gameObject.SetActive(showInfo);
        }
    }
}
