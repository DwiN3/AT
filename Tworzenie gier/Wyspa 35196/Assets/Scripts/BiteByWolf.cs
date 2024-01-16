using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class BiteByWolf : MonoBehaviour
{
    GameObject _player;
    public Text textHints;

    void Start()
    {
        _player = GameObject.FindGameObjectWithTag("Player");
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject == _player)
        {
            textHints.text = "Wilk ciê ugryz³";
            Invoke("SwitchScene", 4.0f);
        }
    }

    void SwitchScene()
    {
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
        SceneManager.LoadScene("Menu");
    }
}
