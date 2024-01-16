using UnityEngine;

public class Running : StateMachineBehaviour
{
    private UnityEngine.AI.NavMeshAgent _nav;
    private Transform _player;
    private float latestDirectionChangeTime;
    public float directionChangeTime = 5f;
    public float WolfVelocity = 50f;
    // OnStateEnter is called when a transition starts and the state machine starts to evaluate this state
    override public void OnStateEnter(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    {
        _player = GameObject.FindGameObjectWithTag("Player").transform;
        _nav = animator.GetComponent<UnityEngine.AI.NavMeshAgent>();
        animator.SetInteger("nextState", 0);
        _nav.ResetPath();
        Vector3 movementDirection = new Vector3(Random.Range(-1.0f, 1.0f), 0, Random.Range(-1.0f,1.0f)).normalized;
        Vector3 movementPerSecond = movementDirection * WolfVelocity;
        Vector3 position = new Vector3(animator.rootPosition.x + (movementPerSecond.x),animator.rootPosition.y, animator.rootPosition.z + (movementPerSecond.z));
        _nav.SetDestination(position);
        latestDirectionChangeTime = Time.time;
    }

    // OnStateUpdate is called on each Update frame between OnStateEnter and OnStateExit callbacks
    override public void OnStateUpdate(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    {
        if (Time.time - latestDirectionChangeTime > directionChangeTime)
        {
            latestDirectionChangeTime = Time.time;
            _nav.ResetPath();
            animator.SetInteger("nextState", Random.Range(1, 4));
        }
    }

    // OnStateExit is called when a transition ends and the state machine finishes evaluating this state
    //override public void OnStateExit(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    //{
    //    
    //}

    // OnStateMove is called right after Animator.OnAnimatorMove()
    //override public void OnStateMove(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    //{
    //    // Implement code that processes and affects root motion
    //}

    // OnStateIK is called right after Animator.OnAnimatorIK()
    //override public void OnStateIK(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    //{
    //    // Implement code that sets up animation IK (inverse kinematics)
    //}
}