using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[RequireComponent(typeof(AudioSource))]

public class CoconutThrower : MonoBehaviour
{
    public AudioClip throwSound;
    public Rigidbody coconutPrefab;
    public float throwSpeed = 30.0f;
    public static bool canThrow = true;
    public int maxCoconuts = 3;


    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        GameObject[] coconuts = GameObject.FindGameObjectsWithTag("coconut");

        if (Input.GetButtonDown("Fire1") && canThrow && coconuts.Length < maxCoconuts)
        {
            GetComponent<AudioSource>().PlayOneShot(throwSound);
            Rigidbody newCoconut = Instantiate(coconutPrefab, transform.position, transform.rotation) as Rigidbody;
            newCoconut.name = "coconut";
            newCoconut.velocity = transform.forward * throwSpeed;
            /*Physics.IgnoreCollision(transform.root.GetComponent<Collider>(), 
            newCoconut.GetComponent<Collider>(), true);*/
        }
    }
}
